import asyncio
import datetime
import json
import logging
import os
import socket
import subprocess
import sys
import traceback

import aioredis

import ray
import ray.gcs_utils
import ray.new_dashboard.modules.reporter.reporter_consts as reporter_consts
import ray.new_dashboard.utils as dashboard_utils
import ray._private.services
import ray.utils
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray.metrics_agent import MetricsAgent, Gauge, Record
import psutil

logger = logging.getLogger(__name__)

try:
    import gpustat.core as gpustat
except ImportError:
    gpustat = None
    logger.warning(
        "Install gpustat with 'pip install gpustat' to enable GPU monitoring.")


def recursive_asdict(o):
    if isinstance(o, tuple) and hasattr(o, "_asdict"):
        return recursive_asdict(o._asdict())

    if isinstance(o, (tuple, list)):
        L = []
        for k in o:
            L.append(recursive_asdict(k))
        return L

    if isinstance(o, dict):
        D = {k: recursive_asdict(v) for k, v in o.items()}
        return D

    return o


def jsonify_asdict(o):
    return json.dumps(dashboard_utils.to_google_style(recursive_asdict(o)))


# A list of gauges to record and export metrics.
METRICS_GAUGES = {
    "toy": Gauge("toy",
                 "just a toy example",
                  "bytes", ["ip"]),
    "node_cpu_utilization": Gauge("node_cpu_utilization",
                                  "Total CPU usage on a ray node",
                                  "percentage", ["ip"]),
    "node_cpu_count": Gauge("node_cpu_count",
                            "Total CPUs available on a ray node", "cores",
                            ["ip"]),
    "node_mem_used": Gauge("node_mem_used", "Memory usage on a ray node",
                           "bytes", ["ip"]),
    "node_mem_available": Gauge("node_mem_available",
                                "Memory available on a ray node", "bytes",
                                ["ip"]),
    "node_mem_total": Gauge("node_mem_total", "Total memory on a ray node",
                            "bytes", ["ip"]),
    "node_mem_utilization": Gauge("node_mem_utilization", "Total memory usage on a ray node",
                            "percentage", ["ip"]),
    "node_gpus_available": Gauge("node_gpus_available",
                                 "Total GPUs available on a ray node",
                                 "percentage", ["ip"]),
    "node_gpus_utilization": Gauge("node_gpus_utilization",
                                   "Total GPUs usage on a ray node",
                                   "percentage", ["ip"]),
    "node_gram_used": Gauge("node_gram_used",
                            "Total GPU RAM usage on a ray node", "bytes",
                            ["ip"]),
    "node_gram_available": Gauge("node_gram_available",
                                 "Total GPU RAM available on a ray node",
                                 "bytes", ["ip"]),
    "node_disk_usage": Gauge("node_disk_usage",
                             "Total disk usage (bytes) on a ray node", "bytes",
                             ["ip"]),
    "node_disk_utilization_percentage": Gauge(
        "node_disk_utilization_percentage",
        "Total disk utilization (percentage) on a ray node", "percentage",
        ["ip"]),
    "node_network_sent": Gauge("node_network_sent", "Total network sent",
                               "bytes", ["ip"]),
    "node_network_received": Gauge("node_network_received",
                                   "Total network received", "bytes", ["ip"]),
    "node_network_send_speed": Gauge(
        "node_network_send_speed", "Network send speed", "bytes/sec", ["ip"]),
    "node_network_receive_speed": Gauge("node_network_receive_speed",
                                        "Network receive speed", "bytes/sec",
                                        ["ip"]),
    "raylet_cpu": Gauge("raylet_cpu", "CPU usage of the raylet on a node.",
                        "percentage", ["ip", "pid"]),
    "raylet_mem": Gauge("raylet_mem", "Memory usage of the raylet on a node",
                        "mb", ["ip", "pid"])
}


class ReporterAgent(dashboard_utils.DashboardAgentModule,
                    reporter_pb2_grpc.ReporterServiceServicer):
    """A monitor process for monitoring Ray nodes.

    Attributes:
        dashboard_agent: The DashboardAgent object contains global config
    """

    def __init__(self, dashboard_agent):

        """Initialize the reporter object."""
        super().__init__(dashboard_agent)


        self._cpu_counts = (psutil.cpu_count(),
                            psutil.cpu_count(logical=False))
        self._ip = ray._private.services.get_node_ip_address()
        self._hostname = socket.gethostname()
        self._workers = set()
        self._network_stats_hist = [(0, (0.0, 0.0))]  # time, (sent, recv)
        self._metrics_agent = MetricsAgent(dashboard_agent.metrics_export_port)
        self._key = f"{reporter_consts.REPORTER_PREFIX}" \
                    f"{self._dashboard_agent.node_id}"

    async def GetProfilingStats(self, request, context):
        pid = request.pid
        duration = request.duration
        profiling_file_path = os.path.join(ray.utils.get_ray_temp_dir(),
                                           f"{pid}_profiling.txt")
        sudo = "sudo" if ray.utils.get_user() != "root" else ""
        process = await asyncio.create_subprocess_shell(
            f"{sudo} $(which py-spy) record "
            f"-o {profiling_file_path} -p {pid} -d {duration} -f speedscope",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            profiling_stats = ""
        else:
            with open(profiling_file_path, "r") as f:
                profiling_stats = f.read()
        return reporter_pb2.GetProfilingStatsReply(
            profiling_stats=profiling_stats, std_out=stdout, std_err=stderr)

    async def ReportOCMetrics(self, request, context):
        # This function receives a GRPC containing OpenCensus (OC) metrics
        # from a Ray process, then exposes those metrics to Prometheus.
        try:
            self._metrics_agent.record_metric_points_from_protobuf(
                request.metrics)
        except Exception:
            logger.error(traceback.format_exc())
        return reporter_pb2.ReportOCMetricsReply()

    @staticmethod
    def _get_cpu_percent():
        return psutil.cpu_percent()

    @staticmethod
    def _get_gpu_usage():
        if gpustat is None:
            return []
        gpu_utilizations = []
        gpus = []
        try:
            gpus = gpustat.new_query().gpus
        except Exception as e:
            logger.debug(f"gpustat failed to retrieve GPU information: {e}")
        for gpu in gpus:
            # Note the keys in this dict have periods which throws
            # off javascript so we change .s to _s
            gpu_data = {
                "_".join(key.split(".")): val
                for key, val in gpu.entry.items()
            }
            gpu_utilizations.append(gpu_data)
        return gpu_utilizations

    @staticmethod
    def _get_boot_time():
        return psutil.boot_time()

    @staticmethod
    def _get_network_stats():
        ifaces = [
            v for k, v in psutil.net_io_counters(pernic=True).items()
            if k[0] == "e"
        ]

        sent = sum((iface.bytes_sent for iface in ifaces))
        recv = sum((iface.bytes_recv for iface in ifaces))
        return sent, recv

    @staticmethod
    def _get_mem_usage():
        total = ray.utils.get_system_memory()
        used = ray.utils.get_used_memory()
        available = total - used
        percent = round(used / total, 3) * 100
        return total, available, percent, used

    @staticmethod
    def _get_disk_usage():
        dirs = [
            os.environ["USERPROFILE"] if sys.platform == "win32" else os.sep,
            ray.utils.get_user_temp_dir(),
        ]
        return {x: psutil.disk_usage(x) for x in dirs}

    def _get_workers(self):
        raylet_proc = self._get_raylet_proc()
        if raylet_proc is None:
            return []
        else:
            workers = set(raylet_proc.children())
            self._workers.intersection_update(workers)
            self._workers.update(workers)
            self._workers.discard(psutil.Process())
            l = [
                w.as_dict(attrs=[
                    "pid",
                    "create_time",
                    "cpu_percent",
                    "cpu_times",
                    "cmdline",
                    "memory_info",
                ]) for w in self._workers if w.status() != psutil.STATUS_ZOMBIE
            ]
            return l

    @staticmethod
    def _get_raylet_proc():
        try:
            curr_proc = psutil.Process()
            # Here, parent is always raylet because the
            # dashboard agent is a child of the raylet process.
            parent = curr_proc.parent()
            if parent is not None:
                if parent.pid == 1:
                    return None
                if parent.status() == psutil.STATUS_ZOMBIE:
                    return None
            return parent
        except (psutil.AccessDenied, ProcessLookupError):
            pass
        return None

    def _get_raylet(self):
        raylet_proc = self._get_raylet_proc()
        if raylet_proc is None:
            return {}
        else:
            return raylet_proc.as_dict(attrs=[
                "pid",
                "create_time",
                "cpu_percent",
                "cpu_times",
                "cmdline",
                "memory_info",
            ])

    def _get_load_avg(self):
        if sys.platform == "win32":
            cpu_percent = psutil.cpu_percent()
            load = (cpu_percent, cpu_percent, cpu_percent)
        else:
            load = os.getloadavg()
        per_cpu_load = tuple((round(x / self._cpu_counts[0], 2) for x in load))
        return load, per_cpu_load

    def _get_all_stats(self):
        now = dashboard_utils.to_posix_time(datetime.datetime.utcnow())
        network_stats = self._get_network_stats()

        self._network_stats_hist.append((now, network_stats))
        self._network_stats_hist = self._network_stats_hist[-7:]
        then, prev_network_stats = self._network_stats_hist[0]
        prev_send, prev_recv = prev_network_stats
        now_send, now_recv = network_stats
        network_speed_stats = ((now_send - prev_send) / (now - then),
                               (now_recv - prev_recv) / (now - then))
        return {
            "now": now,
            "hostname": self._hostname,
            "ip": self._ip,
            "cpu": self._get_cpu_percent(),
            "cpus": self._cpu_counts,
            "mem": self._get_mem_usage(),
            "workers": self._get_workers(),
            "raylet": self._get_raylet(),
            "bootTime": self._get_boot_time(),
            "loadAvg": self._get_load_avg(),
            "disk": self._get_disk_usage(),
            "gpus": self._get_gpu_usage(),
            "network": network_stats,
            "network_speed": network_speed_stats,
            # Deprecated field, should be removed with frontend.
            "cmdline": self._get_raylet().get("cmdline", []),
        }

    @staticmethod
    def _record_stats(stats):
        records_reported = []

        stats['toy'] = 1200.0
        #print(stats)
        #print("Reporter agent record stats!")

        ip = stats["ip"]

        toy = Record(
            gauge=METRICS_GAUGES["toy"],
            value=float(1200),
            tags={"ip": ip})

        # -- CPU per node --
        cpu_usage = float(stats["cpu"])
        cpu_record = Record(
            gauge=METRICS_GAUGES["node_cpu_utilization"],
            value=cpu_usage,
            tags={"ip": ip})

        cpu_count, _ = stats["cpus"]
        cpu_count_record = Record(
            gauge=METRICS_GAUGES["node_cpu_count"],
            value=cpu_count,
            tags={"ip": ip})

        # -- Mem per node --
        mem_total, mem_available, _, mem_used = stats["mem"]
        mem_used_percentage = (mem_used * 100.0)/mem_total

        mem_used_record = Record(
            gauge=METRICS_GAUGES["node_mem_used"],
            value=mem_used,
            tags={"ip": ip})
        mem_utilization_record = Record(
            gauge=METRICS_GAUGES["node_mem_utilization"],
            value=mem_used_percentage,
            tags={"ip": ip})
        mem_available_record = Record(
            gauge=METRICS_GAUGES["node_mem_available"],
            value=mem_available,
            tags={"ip": ip})
        mem_total_record = Record(
            gauge=METRICS_GAUGES["node_mem_total"],
            value=mem_total,
            tags={"ip": ip})

        # -- GPU per node --
        gpus = stats["gpus"]
        gpus_available = len(gpus)

        if gpus_available:
            gpus_utilization, gram_used, gram_total = 0, 0, 0
            for gpu in gpus:
                gpus_utilization += gpu["utilization_gpu"]
                gram_used += gpu["memory_used"]
                gram_total += gpu["memory_total"]

            gram_available = gram_total - gram_used

            gpus_available_record = Record(
                gauge=METRICS_GAUGES["node_gpus_available"],
                value=gpus_available,
                tags={"ip": ip})
            gpus_utilization_record = Record(
                gauge=METRICS_GAUGES["node_gpus_utilization"],
                value=gpus_utilization,
                tags={"ip": ip})
            gram_used_record = Record(
                gauge=METRICS_GAUGES["node_gram_used"],
                value=gram_used,
                tags={"ip": ip})
            gram_available_record = Record(
                gauge=METRICS_GAUGES["node_gram_available"],
                value=gram_available,
                tags={"ip": ip})
            records_reported.extend([
                gpus_available_record, gpus_utilization_record,
                gram_used_record, gram_available_record
            ])

        # -- Disk per node --
        used, free = 0, 0
        for entry in stats["disk"].values():
            used += entry.used
            free += entry.free
        disk_utilization = float(used / (used + free)) * 100
        disk_usage_record = Record(
            gauge=METRICS_GAUGES["node_disk_usage"],
            value=used,
            tags={"ip": ip})
        disk_utilization_percentage_record = Record(
            gauge=METRICS_GAUGES["node_disk_utilization_percentage"],
            value=disk_utilization,
            tags={"ip": ip})

        # -- Network speed (send/receive) stats per node --
        network_stats = stats["network"]
        network_sent_record = Record(
            gauge=METRICS_GAUGES["node_network_sent"],
            value=network_stats[0],
            tags={"ip": ip})
        network_received_record = Record(
            gauge=METRICS_GAUGES["node_network_received"],
            value=network_stats[1],
            tags={"ip": ip})

        # -- Network speed (send/receive) per node --
        network_speed_stats = stats["network_speed"]
        network_send_speed_record = Record(
            gauge=METRICS_GAUGES["node_network_send_speed"],
            value=network_speed_stats[0],
            tags={"ip": ip})
        network_receive_speed_record = Record(
            gauge=METRICS_GAUGES["node_network_receive_speed"],
            value=network_speed_stats[1],
            tags={"ip": ip})

        raylet_stats = stats["raylet"]
        if raylet_stats:
            raylet_pid = str(raylet_stats["pid"])
            # -- raylet CPU --
            raylet_cpu_usage = float(raylet_stats["cpu_percent"]) * 100
            raylet_cpu_record = Record(
                gauge=METRICS_GAUGES["raylet_cpu"],
                value=raylet_cpu_usage,
                tags={
                    "ip": ip,
                    "pid": raylet_pid
                })

            # -- raylet mem --
            raylet_mem_usage = float(raylet_stats["memory_info"].rss) / 1e6
            raylet_mem_record = Record(
                gauge=METRICS_GAUGES["raylet_mem"],
                value=raylet_mem_usage,
                tags={
                    "ip": ip,
                    "pid": raylet_pid
                })
            records_reported.extend([raylet_cpu_record, raylet_mem_record])

        records_reported.extend([
            toy, cpu_record, cpu_count_record, mem_used_record,
            mem_available_record, mem_total_record, mem_utilization_record, disk_usage_record,
            disk_utilization_percentage_record, network_sent_record,
            network_received_record, network_send_speed_record,
            network_receive_speed_record
        ])
        return records_reported

    async def _perform_iteration(self, aioredis_client):
        """Get any changes to the log files and push updates to Redis."""
        while True:
            try:
                stats = self._get_all_stats()
                records_reported = self._record_stats(stats)
                self._metrics_agent.record_reporter_stats(records_reported)
                await aioredis_client.publish(self._key, jsonify_asdict(stats))
            except Exception:
                logger.exception("Error publishing node physical stats.")
            await asyncio.sleep(
                reporter_consts.REPORTER_UPDATE_INTERVAL_MS / 1000)

    async def run(self, server):
        print("Inside run!!")
        aioredis_client = await aioredis.create_redis_pool(
            address=self._dashboard_agent.redis_address,
            password=self._dashboard_agent.redis_password)
        reporter_pb2_grpc.add_ReporterServiceServicer_to_server(self, server)
        await self._perform_iteration(aioredis_client)

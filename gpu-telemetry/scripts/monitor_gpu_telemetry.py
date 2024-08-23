import csv
from pathlib import Path
from queue import Empty
from queue import Queue
import subprocess
from threading import Thread
import time
from typing import Any
from typing import IO
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from absl import app
from absl import flags
from absl import logging


def enqueue_output(out: Optional[IO[Any]], queue: List[str]):
  for line in iter(out.readline, b''):
    queue.put(line)
  out.close()


class Metric(object):
  """Class to represent a GPU telemetry metric."""

  def __init__(self, name: str, data_type: str, unit: str, metric_name: str):
    self._name = name
    self._data_type = data_type
    self._unit = unit
    self._metric_name = metric_name

  def get_data_type(self):
    return self._data_type

  def get_metric_name(self):
    return self._metric_name

  def emit(self, value: Union[float, bool]):
    if self._data_type == 'BOOL':
      return f'{value}', f'{self._name}: {value}'
    elif self._data_type == 'DOUBLE':
      return f'{value:0.2f}', f'{self._name}: {value:0.2f} {self._unit}'
    else:
      return f'{value}', f'{self._name}: {value} {self._unit}'


class GPUTelemetryMonitor:
  """Class to monitor GPU telemetry."""

  def __init__(self, lms: int = 1000, to_csv: bool = False, vm_id: str = ''):
    super().__init__()
    self._to_csv = to_csv
    self._vm_id = vm_id

    # Ensure that persistence mode is enabled
    subprocess.run(
        ['nvidia-smi', '-pm', '1'],
        shell=False,
        capture_output=True,
        check=True,
        text=True,
    )

    cmd = [
        'nvidia-smi',
        '--query-gpu',
        'timestamp,serial,pci.bus_id,temperature.gpu,utilization.gpu,temperature.memory,utilization.memory,memory.used,clocks.gr,clocks.sm,clocks.mem,power.draw.instant,power.draw.average,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,ecc.errors.corrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total',
        '--format',
        'csv,nounits',
        '-lms',
        f'{lms}'
    ]

    self._nvidia_smi_proc = subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    header_line = self._nvidia_smi_proc.stdout.readline().strip()
    self._headers = {
        header: i for i, header in enumerate(header_line.split(', '))
    }
    self.create_metrics()
    self._q = Queue()
    self._queue_thread = Thread(
        target=enqueue_output, args=(self._nvidia_smi_proc.stdout, self._q)
    )
    self._queue_thread.daemon = True  # thread dies with the program
    self._queue_thread.start()

  def terminate(self):
    self._nvidia_smi_proc.terminate()

  def create_metrics(self):
    """Initializes GPU telemetry Metric classes."""

    self._metrics = []
    headers = iter(list(self._headers.keys())[3:])

    # GPU Metrics
    self._metrics.append(
        Metric('gpu/temp', 'INT', 'C', next(headers))
    )
    self._metrics.append(
        Metric('gpu/util', 'INT', '%', next(headers))
    )

    # Memory Metrics
    self._metrics.append(
        Metric('mem/temp', 'INT', 'C', next(headers))
    )
    self._metrics.append(
        Metric('mem/util', 'INT', '%', next(headers))
    )
    self._metrics.append(
        Metric('mem/used', 'INT', 'MB', next(headers))
    )

    # Clock Metrics
    self._metrics.append(
        Metric('clocks/gr', 'INT', 'MHz', next(headers))
    )

    self._metrics.append(
        Metric('clocks/sm', 'INT', 'MHz', next(headers))
    )
    self._metrics.append(
        Metric('clocks/mem', 'INT', 'MHz', next(headers))
    )

    # Power Metrics
    self._metrics.append(
        Metric('power/instant', 'DOUBLE', 'W', next(headers))
    )
    self._metrics.append(
        Metric('power/average', 'DOUBLE', 'W', next(headers))
    )

    # Throttling Metrics
    self._metrics.append(
        Metric('slowdown/hw-thermal', 'BOOL', '', next(headers))
    )
    self._metrics.append(
        Metric('slowdown/hw-power-brake', 'BOOL', '', next(headers))
    )
    self._metrics.append(
        Metric('slowdown/sw-thermal', 'BOOL', '', next(headers))
    )
    self._metrics.append(
        Metric('slowdown/sw-power-cap', 'BOOL', '', next(headers))
    )

    # ECC Metrics
    self._metrics.append(
        Metric('ecc/corrected/volatile', 'INT', '', next(headers))
    )
    self._metrics.append(
        Metric('ecc/corrected/aggregate', 'INT', '', next(headers))
    )
    self._metrics.append(
        Metric('ecc/uncorrected/volatile', 'INT', '', next(headers))
    )
    self._metrics.append(
        Metric('ecc/uncorrected/aggregate', 'INT', '', next(headers))
    )

  def collect_samples(self):
    """Collects target metric samples using nvidia-smi."""

    splits_to_process = []
    csvfile = None
    writer = None
    if self._to_csv:
      csvfile = open('gpu_telemetry.csv', 'w', newline='')
      writer = csv.writer(csvfile, lineterminator='\n')
      writer.writerow(self._headers)

    while self._nvidia_smi_proc.poll() is None:
      try:
        next_line = self._q.get_nowait().strip()
      except Empty:
        break
      splits = next_line.split(', ')
      splits_to_process.append(splits)

    for splits in splits_to_process:
      timestamp = splits[self._headers['timestamp']]
      serial = splits[self._headers['serial']]
      pci_bus_id = splits[self._headers['pci.bus_id']]
      csv_row = [timestamp, serial, pci_bus_id, self._vm_id]
      logging_row = [
          f'timestamp: {timestamp}',
          f'serial: {serial}',
          f'pci_bus_id: {pci_bus_id}',
          f'vm_id: {self._vm_id}',
      ]

      for metric in self._metrics:
        if metric.get_data_type() == 'INT':
          load = int(splits[self._headers[metric.get_metric_name()]])
          csv_out, logging_out = metric.emit(load)
          csv_row.append(csv_out)
          logging_row.append(logging_out)

        elif metric.get_data_type() == 'DOUBLE':
          load = float(splits[self._headers[metric.get_metric_name()]])
          csv_out, logging_out = metric.emit(load)
          csv_row.append(csv_out)
          logging_row.append(logging_out)

        else:
          is_active = (
              True
              if splits[self._headers[metric.get_metric_name()]] == 'Active'
              else False
          )
          csv_out, logging_out = metric.emit(is_active)
          csv_row.append(csv_out)
          logging_row.append(logging_out)

      logging.info(', '.join(logging_row))
      if self._to_csv:
        writer.writerow(csv_row)

    if self._to_csv:
      csvfile.close()


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'lms',
    1000,
    'Continuously report query data at the specified interval in miliseconds.',
)
flags.DEFINE_bool('to_csv', False, 'Output to a CSV file.')
flags.DEFINE_string('vm_id', '', 'VM ID.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gpu_monitor = GPUTelemetryMonitor(
      lms=FLAGS.lms, to_csv=FLAGS.to_csv, vm_id=FLAGS.vm_id
  )
  termination_path = Path('/usr/share/telemetry/workload_terminated')

  while not termination_path.exists():
    time.sleep(5)
    gpu_monitor.collect_samples()


if __name__ == '__main__':
  app.run(main)
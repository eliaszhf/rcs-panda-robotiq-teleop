#!/usr/bin/env bash

set -euo pipefail

robot_ip="${1:-192.168.178.12}"
irq_cpu="${2:-5}"
if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo: sudo $0 $robot_ip $irq_cpu" >&2
  exit 1
fi

if [[ ! "$irq_cpu" =~ ^[0-9]+$ ]] || [[ ! -d "/sys/devices/system/cpu/cpu$irq_cpu" ]]; then
  echo "Invalid IRQ CPU: $irq_cpu" >&2
  exit 1
fi

interface="$(ip route get "$robot_ip" | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}')"

if [[ -z "$interface" ]]; then
  echo "Cannot determine the network interface for $robot_ip" >&2
  exit 1
fi

governor_count=0
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  available="${governor%/scaling_governor}/scaling_available_governors"
  if [[ ! -r "$available" ]] || ! grep -qw performance "$available"; then
    echo "CPU governor 'performance' is unavailable for $governor" >&2
    exit 1
  fi
  printf '%s\n' performance >"$governor"
  governor_count=$((governor_count + 1))
done
if [[ $governor_count -eq 0 ]]; then
  echo "No CPU frequency governor controls were found" >&2
  exit 1
fi

ethtool -K "$interface" gro off

# Keep the 1 kHz FCI receive path away from CPU0 housekeeping. irqbalance may
# otherwise move the NIC interrupt back while an episode is running. CPU5 is
# the quiet logical CPU selected for this 12-thread lab workstation; an
# explicit second argument can override it on another machine.
irq="$(awk -v interface="$interface" '$0 ~ interface {gsub(":", "", $1); print $1; exit}' /proc/interrupts)"
if [[ -z "$irq" ]] || [[ ! -w "/proc/irq/$irq/smp_affinity_list" ]]; then
  echo "Cannot find a writable IRQ affinity for $interface" >&2
  exit 1
fi
if systemctl is-active --quiet irqbalance; then
  systemctl stop irqbalance
fi
printf '%s\n' "$irq_cpu" >"/proc/irq/$irq/smp_affinity_list"

echo "Applied transient libfranka tuning for $robot_ip on $interface"
echo "CPU governors:"
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  printf '  %s: ' "$(basename "$(dirname "$(dirname "$governor")")")"
  cat "$governor"
done
echo "Network GRO:"
ethtool -k "$interface" | awk '/generic-receive-offload:/ {print "  " $0}'
echo "Network IRQ:"
echo "  IRQ $irq pinned to CPU $(cat "/proc/irq/$irq/effective_affinity_list")"
echo "  irqbalance: $(systemctl is-active irqbalance || true)"
echo "These settings normally revert after reboot."

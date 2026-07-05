---
name: quest-teleop
description: Use when working with Meta/Oculus Quest 3 or Quest headsets for dex-teleop input, including ADB setup, USB or wireless connection, finding the headset IP, running scripts/test_oculus_reader.py, checking Oculus Reader transforms/buttons, and troubleshooting missing USB debugging prompts, unauthorized/offline devices, or wireless ADB failures.
---

# Quest Teleop

Use this skill to connect a Quest/Quest 3 headset to the `dex-teleop` repo and verify Oculus Reader data.

## Environment

Always use the repo conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
```

The diagnostic script is:

```bash
python scripts/test_oculus_reader.py --duration 5
```

For wireless mode, pass `--ip <quest-ip>`.

## USB ADB setup

1. Confirm Quest developer mode is enabled in the Meta Horizon / Quest phone app.
2. Connect the headset with a USB-C cable that supports data transfer.
3. Put on the headset and accept the USB debugging prompt.
   - Select “Always allow from this computer” when available.
   - Quest settings may show a Developer menu without a separate “USB Debugging” toggle; the important step is accepting the prompt after USB connection.
4. Check ADB:

```bash
adb devices -l
```

Expected:

```text
<serial>    device ... model:Quest_3 ...
```

If the list is empty, check cable, USB port, headset wake/unlock state, and developer mode. If it says `unauthorized`, put on the headset and accept the debugging prompt. If it says `offline`, restart ADB and reconnect:

```bash
adb kill-server
adb start-server
adb devices -l
```

## Run Oculus Reader over USB

Run from the repo root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
python scripts/test_oculus_reader.py --duration 5
```

Success looks like:

```text
[PASS] Received Oculus Reader data.
[PASS] Last sample: transforms=['l', 'r'], buttons=[...]
```

`transforms=['l', 'r']` means left and right controller poses are being read. Buttons include keys such as `A`, `B`, `X`, `Y`, `leftGrip`, `rightGrip`, `leftJS`, and `rightJS`.

## Find the Quest IP

With USB ADB connected:

```bash
adb shell ip route
adb shell ip addr show wlan0
```

Use the `src` address from `ip route` or the `inet` address under `wlan0`.

Example from a working Quest 3 session:

```text
10.42.0.0/24 dev wlan0 proto kernel scope link src 10.42.0.61
```

The Quest IP is `10.42.0.61` in that example.

## Wireless ADB setup

Keep USB connected for the first switch to TCP mode:

```bash
adb tcpip 5555
adb connect <quest-ip>:5555
adb devices -l
```

Expected:

```text
<quest-ip>:5555    device ... model:Quest_3 ...
```

Then wireless Oculus Reader test:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex-teleop
export PYTHONNOUSERSITE=1
python scripts/test_oculus_reader.py --ip <quest-ip> --duration 5
```

After `adb devices -l` shows `<quest-ip>:5555 device`, USB can usually be unplugged.

## Wireless troubleshooting

- If `adb connect <old-ip>:5555` says `No route to host`, re-check the current IP with `adb shell ip route` or `adb devices -l`; the Quest may have moved networks.
- If `adb tcpip 5555` makes the USB device disappear or go offline, run `adb devices -l` again. Sometimes the device immediately reappears as `<quest-ip>:5555`.
- If multiple devices are listed, target one explicitly:

```bash
adb -s <serial-or-ip:5555> shell ip route
```

- If wireless ADB is connected but Oculus Reader fails, verify the headset and computer are on the same reachable network and rerun:

```bash
adb devices -l
python scripts/test_oculus_reader.py --ip <quest-ip> --duration 8 --interval 0.5
```

## Known working result in this repo

On 2026-07-05, Quest 3 worked in both USB and wireless ADB modes. The successful wireless device address was:

```text
10.42.0.61:5555    device product:eureka model:Quest_3 device:eureka
```

The diagnostic script received:

```text
transforms=['l', 'r']
```

and button data including grip, trigger, joystick, and face-button fields.

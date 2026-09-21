import rcs_robotiq2f85.hw as hw


class FakeRobotiq2F85:
    def __init__(self, serial_number: str, async_control: bool):
        self.serial_number = serial_number
        self.async_control = async_control
        self.reset_calls = 0
        self.moves = []
        self.closed = False
        self.tty_device = hw.LinuxFindTTYWithSerialNumber().find(serial_number)

    def reset(self):
        self.reset_calls += 1

    def go_to(self, *, opening: float, speed: float, force: float):
        self.moves.append((opening, speed, force))

    def close(self):
        self.closed = True


def test_wrapper_uses_robotiq2f_020_api_without_opening_real_serial(monkeypatch):
    monkeypatch.setattr(hw, "Robotiq2F85", FakeRobotiq2F85)
    cfg = hw.RobotiQ2F85GripperConfig(
        serial_number="TEST-SERIAL",
        speed=20.0,
        force=20.0,
        async_control=True,
        serial_device="/dev/ttyUSB1",
    )

    gripper = hw.RobotiQ2F85Gripper(cfg)
    assert gripper.gripper.serial_number == "TEST-SERIAL"
    assert gripper.gripper.async_control is True
    assert gripper.gripper.tty_device == "/dev/ttyUSB1"
    assert gripper.gripper.reset_calls == 1

    gripper.set_normalized_width(0.5)
    assert gripper.gripper.moves == [(42.5, 20.0, 20.0)]
    assert gripper.get_normalized_width() == 0.5

    gripper.close()
    assert gripper.gripper.closed is True

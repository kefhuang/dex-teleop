class RobotError(RuntimeError): pass
class RobotConnectionError(RobotError): pass
class RobotStateUnavailable(RobotError): pass
class RobotStateStale(RobotError): pass
class RobotCommandError(RobotError): pass
class IKError(RobotError): pass
class IKUnreachableError(IKError): pass

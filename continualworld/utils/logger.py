from abc import abstractmethod, ABC

class Logger(ABC):
    def __init__(self) -> None:
        self._timestep: int = 0

    def increase_timestep(self) -> None:
        self._timestep += 1

    def set_timestep(self, timestep: int) -> None:
        self._timestep = timestep

    @abstractmethod
    def log(self, metric: str, value: float) -> None:
        ...


class TerminalLogger(Logger):
    def log(self, metric: str, value: float) -> None:
        print(f'[{self._timestep}] {metric}: {value:.2f}')

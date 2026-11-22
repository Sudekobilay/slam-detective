"""One command/measurement loop, independent of CLI and persistence."""
from dataclasses import dataclass
import math
import time


@dataclass
class EpisodeResult:
    agent: object
    events: list
    commands: list
    records: dict
    stop_reason: str | None
    runtime_s: float

    def legacy(self):
        return self.agent, self.events, self.commands, self.records, self.stop_reason


def run_episode(world, agent, *, minimum_horizon_s=4., changed_commands=True):
    if not math.isfinite(minimum_horizon_s) or minimum_horizon_s <= 0:
        raise ValueError('Positive finite horizon required.')
    events = world.initial_measurements()
    agent.process(events)
    commands, reason = [], None
    deadline = minimum_horizon_s
    tick = time.perf_counter()
    while agent.time_s < deadline - 1e-9:
        rates = agent.command()
        command = {'interval_start_s': agent.time_s, 'left_rate_rad_s': rates[0],
                   'right_rate_rad_s': rates[1], 'mode': agent.mode, 'applied': False}
        if changed_commands:
            command['changed_from_normal'] = rates != agent.normal_command
        commands.append(command)
        try:
            batch = world.step(*rates)
        except RuntimeError as error:
            reason = str(error)
            break
        command['applied'] = True
        agent.process(batch)
        events.extend(batch)
        deadline = max(minimum_horizon_s, agent.window_end_s or minimum_horizon_s)
    return EpisodeResult(agent, events, commands, world.evaluation_records(), reason,
                         time.perf_counter() - tick)

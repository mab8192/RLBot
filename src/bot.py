from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo, BallInfo

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

# A collection of the things we care about for decision making
@dataclass
class State:
    me: PlayerInfo
    teammates: List[PlayerInfo]
    enemies: List[PlayerInfo]
    ball: BallInfo
    seconds_elapsed: float

    def parse_packet(self, packet: GameTickPacket):
        """Update all of this State's fields from the given packet"""
        self.ball = packet.game_ball
        self.seconds_elapsed = packet.game_info.seconds_elapsed

class Action(Enum):
    WAIT = 1 # Wait in your current location
    BALLCHASE = 2 # Chase down the ball and attempt to hit it forward with power
    DEMO = 3 # Chase down an enemy car and attempt to demo them
    SHOOT = 4 # Shoot the ball on target, usually with a flip
    PASS = 5 # Pass the ball to a teammate
    DEFEND = 6 # Hit the ball away from the center of the field, preferably not in your own net
    POSITION_DEFENSE = 7 # Position behind a teammate, whereever we expect the ball to go
    POSITION_OFFENSE = 8 # Position next to or in front of a teammate
    GET_BOOST = 9 # Move to the nearest boost pad

    def __str__(self):
        match self:
            case Action.WAIT: return "WAIT"
            case Action.BALLCHASE: return "BALLCHASE"
            case Action.DEMO: return "DEMO"
            case Action.SHOOT: return "SHOOT"
            case Action.PASS: return "PASS"
            case Action.DEFEND: return "DEFEND"
            case Action.POSITION_DEFENSE: return "POSITION_DEFENSE"
            case Action.POSITION_OFFENSE: return "POSITION_OFFENSE"
            case Action.GET_BOOST: return "GET_BOOST"
            case _: raise NotImplementedError("Invalid action: " + self.action)

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.state: State = State()
        self.packet: GameTickPacket = None
        self.action: Action = None
        self.target_location: Vec3 = None

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """
        self.packet = packet
        self.state.parse_packet(packet)

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        self.action = self.decide()

        match self.action:
            case Action.WAIT: return self.wait()
            case Action.BALLCHASE: return self.ballchase()
            case Action.DEMO: return self.demo()
            case Action.SHOOT: return self.shoot()
            case Action.PASS: return self.pass_to_player(None)
            case Action.DEFEND: return self.defend()
            case Action.POSITION_DEFENSE: return self.position_defense()
            case Action.POSITION_OFFENSE: return self.position_offense()
            case Action.GET_BOOST: return self.get_boost()
            case _: raise NotImplementedError("Invalid action: " + self.action)

    def render_debug(self):
        # Draw some things to help understand what the bot is thinking
        my_location = Vec3(self.state.me.physics.location)
        my_velocity = Vec3(self.state.me.physics.velocity)
        self.renderer.draw_string_2d(0, 0, 10, 10, str(self.action), self.renderer.white())
        self.renderer.draw_line_3d(my_location, self.target_location, self.renderer.white())
        self.renderer.draw_string_3d(my_location, 1, 1, f'Speed: {my_velocity.length():.1f}', self.renderer.white())
        self.renderer.draw_rect_3d(self.target_location, 8, 8, True, self.renderer.cyan(), centered=True)

    def front_flip(self, packet) -> SimpleControllerState:
        # Send some quickchat just for fun
        self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True, throttle=1.0)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False, throttle=1.0)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1, throttle=1.0)),
            ControlStep(duration=0.8, controls=SimpleControllerState(throttle=1.0)),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    def move_to_target(self, target_location):
        controls = SimpleControllerState()
        controls.steer = steer_toward_target(self.state.me, target_location)
        controls.throttle = 1.0
        controls.boost = True

        return controls

    def decide(self) -> Action:
        return Action.BALLCHASE

    def wait(self) -> SimpleControllerState:
        return SimpleControllerState()

    def ballchase(self) -> SimpleControllerState:
        my_location = Vec3(self.state.me.physics.location)
        my_velocity = Vec3(self.state.me.physics.velocity)
        ball_location = Vec3(self.state.ball.physics.location)
        target_location = ball_location

        if my_location.dist(ball_location) > 1500:
            # We're far away from the ball, let's try to lead it a little bit
            ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
            ball_in_future = find_slice_at_time(ball_prediction, self.state.seconds_elapsed + 2)

            # ball_in_future might be None if we don't have an adequate ball prediction right now, like during
            # replays, so check it to avoid errors.
            if ball_in_future is not None:
                target_location = Vec3(ball_in_future.physics.location)

        if 750 < my_velocity.length() < 800:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.front_flip(self.packet)

        return self.move_to_target(target_location)

    def pass_to_player(self, player):
        return SimpleControllerState()

    def demo(self, player):
        return SimpleControllerState()

    def shoot(self):
        return SimpleControllerState()

    def defend(self):
        return SimpleControllerState()

    def position_defense(self):
        return SimpleControllerState()

    def position_offense(self):
        return SimpleControllerState()

    def get_boost(self):
        return SimpleControllerState()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:48:17 2023

@author: oscar
"""

import pygame
from pynput.keyboard import Key, Listener

class Controller:

    def __init__(self, id, dead_zone = 0.15):
        """
        Initializes a controller.

        Args:
            id: The ID of the controller which must be a value from `0` to
                `pygame.joystick.get_count() - 1`
            dead_zone: The size of dead zone for the    analog sticks (default 0.15)
        """

        self._joystick = pygame.joystick.Joystick(id)
        self._joystick.init()
        self.dead_zone = dead_zone
        
        self.listener = Listener(on_press=self.press)
        self.listener.start()

        self._steer_idx     = 0
        self._clutch_idx  = 1
        self._throttle_idx = 2
        self._brake_idx     = 3
        self._reverse_idx   = 4
        self._handbrake_idx = 5

        self.corner_flag = False
        self.abandon_flag = False

    def press(self, key):
        if key == Key.left:
            self.corner_flag = True
        elif key == Key.right:
            self.abandon_flag = True

    def get_id(self):
        """
        Returns:
            The ID of the controller. This is the same as the ID passed into
            the initializer.
        """

        return self._joystick.get_id()

    def get_buttons(self):
        """
        Gets the state of each button on the controller.

        Returns:
            A tuple with the state of each button. 1 is pressed, 0 is unpressed.
        """

        numButtons = self._joystick.get_numbuttons()
        jsButtons = [float(self._joystick.get_button(i)) for i in range(numButtons)]

        return (jsButtons)


    def get_axis(self):
        """
        Gets the state of each axis on the controller.

        Returns:
            The axes values x as a tuple such that

            -1 <= x <= 1

        """

        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        return (jsInputs)


    def get_steer(self):
        """
        Gets the state of the steering wheel.

        Returns:
            A value x such that

            -1 <= x <= 1 && -1 <= y <= 1

            Negative values are left.
            Positive values are right.
        """


        return (self.get_axis()[self._steer_idx])


    def get_clutch(self):
        """
        Gets the state of the gear pedal.

        Returns:
            A value x such that

            -1 <= x <= 1

        """


        return (self.get_axis()[self._clutch_idx])


    def get_break(self):
        """
        Gets the state of the break pedal.

        Returns:
            A value x such that

            -1 <= x <= 1

        """


        return (self.get_axis()[self._brake_idx])



    def get_throttle(self):
        """
        Gets the state of the throttle pedal.

        Returns:
            A value x such that

            -1 <= x <= 1

        """


        return (self.get_axis()[self._throttle_idx])


    def get_reverse(self):
        """
        Gets the state of the reverse button.

        Returns:
            A value x such that 1 is pressed, 0 is unpressed.

        """


        return (self.get_buttons()[self._reverse_idx])


    def get_handbrake(self):
        """
        Gets the state of the handbrake.

        Returns:
            A value x such that 1 is pressed, 0 is unpressed.
        """


        return (self.get_buttons()[self._handbrake_idx])
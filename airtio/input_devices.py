import uinput
import logging
import numpy as np


def setup_input_devices():
    """Set up virtual keyboard, mouse, and Xbox One controller using uinput."""
    try:
        # Full keyboard: all standard keys
        keyboard_events = [getattr(uinput, f"KEY_{key}") for key in (
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
            "1 2 3 4 5 6 7 8 9 0 "
            "ENTER ESCAPE BACKSPACE TAB SPACE MINUS EQUAL LEFTBRACKET RIGHTBRACKET "
            "BACKSLASH SEMICOLON QUOTE GRAVE COMMA DOT SLASH CAPSLOCK "
            "F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 "
            "LEFTCTRL LEFTSHIFT LEFTALT RIGHTCTRL RIGHTSHIFT RIGHTALT "
            "UP DOWN LEFT RIGHT").split()]
        keyboard = uinput.Device(keyboard_events, name="virtual-keyboard")

        # Mouse: left/right buttons, movement, scroll
        mouse_events = [
            uinput.BTN_LEFT,
            uinput.BTN_RIGHT,
            uinput.REL_X,
            uinput.REL_Y,
            uinput.REL_WHEEL,
        ]
        mouse = uinput.Device(mouse_events, name="virtual-mouse")

        # Xbox One controller: buttons, D-pad, sticks, triggers
        gamepad_events = [
            uinput.BTN_A, uinput.BTN_B, uinput.BTN_X, uinput.BTN_Y,
            uinput.BTN_TL, uinput.BTN_TR,  # LB, RB
            uinput.BTN_START, uinput.BTN_SELECT, uinput.BTN_MODE,  # Start, Back, Guide
            uinput.BTN_DPAD_UP, uinput.BTN_DPAD_DOWN, uinput.BTN_DPAD_LEFT, uinput.BTN_DPAD_RIGHT,
            uinput.ABS_X + (-32768, 32767, 0, 0),  # Left stick X
            uinput.ABS_Y + (-32768, 32767, 0, 0),  # Left stick Y
            uinput.ABS_RX + (-32768, 32767, 0, 0),  # Right stick X
            uinput.ABS_RY + (-32768, 32767, 0, 0),  # Right stick Y
            uinput.ABS_Z + (0, 32767, 0, 0),  # Left trigger
            uinput.ABS_RZ + (0, 32767, 0, 0),  # Right trigger
        ]
        gamepad = uinput.Device(gamepad_events, name="virtual-xbox-one")

        logging.info("Virtual input devices created: keyboard, mouse, Xbox One controller")
        return keyboard, mouse, gamepad
    except Exception as e:
        logging.error(f"Failed to create input devices: {str(e)}")
        raise


def process_keyboard_array(arr, keyboard):
    """Process numpy array for keyboard actions."""
    if len(arr) != 36:  # Match action map size
        logging.error(f"Keyboard array size {len(arr)} does not match expected 36")
        return

    action_map = {
        0: (uinput.KEY_A, "Keyboard A press"),
        1: (uinput.KEY_B, "Keyboard B press"),
        2: (uinput.KEY_C, "Keyboard C press"),
        3: (uinput.KEY_D, "Keyboard D press"),
        4: (uinput.KEY_E, "Keyboard E press"),
        5: (uinput.KEY_F, "Keyboard F press"),
        6: (uinput.KEY_G, "Keyboard G press"),
        7: (uinput.KEY_H, "Keyboard H press"),
        8: (uinput.KEY_I, "Keyboard I press"),
        9: (uinput.KEY_J, "Keyboard J press"),
        10: (uinput.KEY_K, "Keyboard K press"),
        11: (uinput.KEY_L, "Keyboard L press"),
        12: (uinput.KEY_M, "Keyboard M press"),
        13: (uinput.KEY_N, "Keyboard N press"),
        14: (uinput.KEY_O, "Keyboard O press"),
        15: (uinput.KEY_P, "Keyboard P press"),
        16: (uinput.KEY_Q, "Keyboard Q press"),
        17: (uinput.KEY_R, "Keyboard R press"),
        18: (uinput.KEY_S, "Keyboard S press"),
        19: (uinput.KEY_T, "Keyboard T press"),
        20: (uinput.KEY_U, "Keyboard U press"),
        21: (uinput.KEY_V, "Keyboard V press"),
        22: (uinput.KEY_W, "Keyboard W press"),
        23: (uinput.KEY_X, "Keyboard X press"),
        24: (uinput.KEY_Y, "Keyboard Y press"),
        25: (uinput.KEY_Z, "Keyboard Z press"),
        26: (uinput.KEY_1, "Keyboard 1 press"),
        27: (uinput.KEY_2, "Keyboard 2 press"),
        28: (uinput.KEY_3, "Keyboard 3 press"),
        29: (uinput.KEY_4, "Keyboard 4 press"),
        30: (uinput.KEY_5, "Keyboard 5 press"),
        31: (uinput.KEY_ENTER, "Keyboard Enter press"),
        32: (uinput.KEY_LEFTCTRL, "Keyboard Left Ctrl press"),
        33: (uinput.KEY_LEFTSHIFT, "Keyboard Left Shift press"),
        34: (uinput.KEY_UP, "Keyboard Up press"),
        35: (uinput.KEY_DOWN, "Keyboard Down press"),
    }

    try:
        for idx, value in enumerate(arr):
            if abs(value) < 1e-6:  # Ignore near-zero values
                continue
            if idx in action_map:
                event, log_message = action_map[idx]
                keyboard.emit(event, 1)
                keyboard.emit(event, 0)
                logging.info(log_message)
    except Exception as e:
        logging.error(f"Failed to process keyboard array: {str(e)}")


def process_mouse_array(arr, mouse):
    """Process numpy array for mouse actions."""
    if len(arr) != 5:  # Match action map size
        logging.error(f"Mouse array size {len(arr)} does not match expected 5")
        return

    action_map = {
        0: (uinput.BTN_LEFT, "Mouse left click"),
        1: (uinput.BTN_RIGHT, "Mouse right click"),
        2: (uinput.REL_X, "Mouse move X"),
        3: (uinput.REL_Y, "Mouse move Y"),
        4: (uinput.REL_WHEEL, "Mouse scroll"),
    }

    try:
        for idx, value in enumerate(arr):
            if abs(value) < 1e-6:  # Ignore near-zero values
                continue
            if idx in action_map:
                event, log_message = action_map[idx]
                if event in (uinput.REL_X, uinput.REL_Y, uinput.REL_WHEEL):
                    # Scale -1 to 1 to -128 to 128
                    scaled_value = int(value * 128)
                    if abs(scaled_value) >= 1:  # Ignore small movements
                        mouse.emit(event, scaled_value)
                        logging.info(f"{log_message}: {scaled_value}")
                else:
                    mouse.emit(event, 1)
                    mouse.emit(event, 0)
                    logging.info(log_message)
    except Exception as e:
        logging.error(f"Failed to process mouse array: {str(e)}")


def process_gamepad_array(arr, gamepad):
    """Process numpy array for Xbox One controller actions."""
    if len(arr) != 17:  # Match action map size
        logging.error(f"Gamepad array size {len(arr)} does not match expected 17")
        return

    action_map = {
        0: (uinput.BTN_A, "Gamepad A press"),
        1: (uinput.BTN_B, "Gamepad B press"),
        2: (uinput.BTN_X, "Gamepad X press"),
        3: (uinput.BTN_Y, "Gamepad Y press"),
        4: (uinput.BTN_TL, "Gamepad LB press"),
        5: (uinput.BTN_TR, "Gamepad RB press"),
        6: (uinput.BTN_START, "Gamepad Start press"),
        7: (uinput.BTN_SELECT, "Gamepad Back press"),
        8: (uinput.BTN_MODE, "Gamepad Guide press"),
        9: (uinput.BTN_DPAD_UP, "Gamepad D-pad Up press"),
        10: (uinput.BTN_DPAD_DOWN, "Gamepad D-pad Down press"),
        11: (uinput.BTN_DPAD_LEFT, "Gamepad D-pad Left press"),
        12: (uinput.BTN_DPAD_RIGHT, "Gamepad D-pad Right press"),
        13: (uinput.ABS_X, "Gamepad left stick X"),
        14: (uinput.ABS_Y, "Gamepad left stick Y"),
        15: (uinput.ABS_RX, "Gamepad right stick X"),
        16: (uinput.ABS_RY, "Gamepad right stick Y"),
        17: (uinput.ABS_Z, "Gamepad left trigger"),
        18: (uinput.ABS_RZ, "Gamepad right trigger"),
    }

    try:
        for idx, value in enumerate(arr):
            if abs(value) < 1e-6:  # Ignore near-zero values
                continue
            if idx in action_map:
                event, log_message = action_map[idx]
                if event in (uinput.ABS_X, uinput.ABS_Y, uinput.ABS_RX, uinput.ABS_RY):
                    # Scale -1 to 1 to -32768 to 32767
                    scaled_value = int(value * 32767)
                    if abs(scaled_value) >= 1:
                        gamepad.emit(event, scaled_value)
                        logging.info(f"{log_message}: {scaled_value}")
                elif event in (uinput.ABS_Z, uinput.ABS_RZ):
                    # Scale 0 to 1 to 0 to 32767 (triggers)
                    scaled_value = int(value * 32767) if value > 0 else 0
                    if abs(scaled_value) >= 1:
                        gamepad.emit(event, scaled_value)
                        logging.info(f"{log_message}: {scaled_value}")
                else:
                    gamepad.emit(event, 1)
                    gamepad.emit(event, 0)
                    logging.info(log_message)
    except Exception as e:
        logging.error(f"Failed to process gamepad array: {str(e)}")
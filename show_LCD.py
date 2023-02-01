import drivers
from time import sleep

def show_LCD():
 lcd = drivers.Lcd()
 lcd.lcd_display_string("Trash", 1)
 sleep(3)
 lcd.lcd_clear()

'''import pywhatkit
import time
pywhatkit.sendwhatmsg('+919786829666', 'hello')
'''

import pywhatkit
from datetime import datetime

# Specify the phone number (with country code) and the message you want to send
phone_number = "+919786829666"
current_time = datetime.now().strftime("%H:%M")
message = f"Hello, this is an automated message. The current time is {current_time}."

# Send the message
pywhatkit.sendwhatmsg(phone_number, message, datetime.now().hour, datetime.now().minute+1)

'''
import pywhatkit

# Specify the phone number (with country code) and the message you want to send
phone_number = "+1234567890"
message = "Hello, this is an automated message. The current time is <CURRENT_TIME>."

# Send the message instantly
pywhatkit.sendwhatmsg_instantly(phone_number, message.replace("<CURRENT_TIME>", pywhatkit.timeConverter(pywhatkit.now())[0]))
'''
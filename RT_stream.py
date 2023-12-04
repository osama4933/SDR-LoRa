import math
import multiprocessing
from multiprocessing import Manager
from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa
import socket

import time
from threading import Thread
from queue import Queue

from matplotlib import pyplot as plt

from Active_Session import Active_Session
from Active_Period_Detector_new import Active_Period_Detector_new
from Active_Period_Detector import Active_Period_Detector2
# from State import State
from utils import *

############################### Variables ###############################
LOGGER_LOC = 'DATA_LOG.txt'
RAW_FS = 500e3					# SDR's raw sampling freq
LORA_BW = 125e3        # LoRa bandwidth
LORA_CHANNELS = [1]  # channels to process

UPSAMPLE_FACTOR = 4             		# factor to downsample to
FS = LORA_BW * UPSAMPLE_FACTOR    		# sampling rate after conversion
BW = LORA_BW                      		# LoRa signal bandwidth
OVERLAP = 10 * UPSAMPLE_FACTOR#int(10 * RAW_FS / BW)
THROTTLE_RATE = 1

##########################################################################

def spawn_a_worker(my_channel, input_queue, output_queue):
    ###########################################################################################################
    # worker = YourDemodulatorImplementation(my_channel, input_queue, output_queue)
    # worker.start_consuming()
    ###########################################################################################################
    a = 1

def IQ_SOURCE(chan, chan_num):
    from client_config import RAW_FS, OVERLAP
    t2 = '127.0.0.1'
    if chan_num == 1:
        port = 4900

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((t2, port))
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)

    recver = list()
    counter = 0
    tmp = np.array([])
    atime = time.time()

    while True:
        message, address = server_socket.recvfrom(2 ** 15)
        if len(message) > 0:
            #print(len(message))
            counter += len(message)
            recver.append(message)

            if counter > int(RAW_FS + OVERLAP * 2) * 8:
                final = np.array(recver).view(dtype=np.complex64)       # flatten
                final = np.concatenate((tmp, final))                    # combine with previous
                tmp = final[int(RAW_FS):]                               # get overlap
                final = final[0:int(RAW_FS + OVERLAP * 2)]              # final chunk
                chan.put((final, time.time()))
                print(f"Received a chunk at time {time.time()}\n")      # Comment this print later
                recver = list()
                counter = len(tmp) * 8
                atime += 1
        else:
            time.sleep(0.1)


big_q = multiprocessing.Queue()
channel_streams = []
if __name__ == "__main__":
    # manager = Manager()

    for i in LORA_CHANNELS:
        in_queue = multiprocessing.Queue()
        channel_streams.append(in_queue)
        multiprocessing.Process(target=spawn_a_worker, args=(i, in_queue, big_q)).start()



    time.sleep(2.0)
    for i in range(len(LORA_CHANNELS)):
        print(LORA_CHANNELS[i])
        multiprocessing.Process(target=IQ_SOURCE, args=(channel_streams[i], LORA_CHANNELS[i])).start()

    time.sleep(7260)
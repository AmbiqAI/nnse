"""
Audio Viewer for the audio data from EVB
"""
import os
import argparse
import sys
import wave
import multiprocessing
from multiprocessing import Process, Array, Lock
import time
import erpc
import GenericDataOperations_EvbToPc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Define the RPC service handlers - one for each EVB-to-PC RPC function
FRAMES_TO_SHOW  = 500
SAMPLING_RATE   = 16000
HOP_SIZE        = 160

class DataServiceClass:
    """
    Capture Audio data: EVB->PC
    """
    def __init__(self, databuf, wavout, lock, is_record, cyc_count):
        self.cyc_count      = cyc_count
        self.wavefile       = None
        self.wavename       = wavout
        self.databuf        = databuf
        self.lock           = lock
        self.is_record      = is_record

    def wavefile_init(self, wavename):
        """
        wavefile initialization
        """
        fldr = 'audio_result'
        os.makedirs(fldr, exist_ok=True)
        wavefile = wave.open(f'{fldr}/{wavename}', 'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(16000)
        return wavefile

    def ns_rpc_data_sendBlockToPC(self, pcmBlock): # pylint: disable=invalid-name
        """
        callback function that data sent from EVB to PC.
        """
        self.lock.acquire()
        is_record = self.is_record[0]
        self.lock.release()
        if is_record == 0:
            if self.wavefile:
                self.wavefile.close()
                self.wavefile = None
                print('Stop recording')
        else:
            # The data 'block' (in C) is defined below:
            # static char msg_store[30] = "Audio16bPCM_to_WAV";

            # // Block sent to PC
            # static dataBlock outBlock = {
            #     .length = SAMPLES_IN_FRAME * sizeof(int16_t),
            #     .dType = uint8_e,
            #     .description = msg_store,
            #     .cmd = write_cmd,
            #     .buffer = {.data = (uint8_t *)in16AudioDataBuffer, // point this to audio buffer # pylint: disable=line-too-long
            #             .dataLength = SAMPLES_IN_FRAME * sizeof(int16_t)}};
            if self.wavefile:
                self.lock.acquire()
                cyc_count = self.cyc_count[0]
                self.lock.release()
            else:
                print('Start recording')
                cyc_count = 0
                self.lock.acquire()
                self.cyc_count[0] = cyc_count
                self.lock.release()
                self.wavefile = self.wavefile_init(self.wavename)

            if (pcmBlock.cmd == GenericDataOperations_EvbToPc.common.command.write_cmd) \
                     and (pcmBlock.description == "Audio16bPCM_to_WAV"):

                self.lock.acquire()
                self.wavefile.writeframesraw(pcmBlock.buffer)
                self.lock.release()

                # Data is a 16 bit PCM sample
                self.lock.acquire()
                fdata = np.frombuffer(pcmBlock.buffer, dtype=np.int16).copy() / 32768.0
                self.lock.release()
                start = cyc_count * HOP_SIZE

                self.lock.acquire()
                self.databuf[start:start+HOP_SIZE] = fdata
                self.lock.release()
                cyc_count = (cyc_count+1) % FRAMES_TO_SHOW
                self.lock.acquire()
                self.cyc_count[0] = cyc_count
                self.lock.release()

        sys.stdout.flush()

        return 0

    def ns_rpc_data_fetchBlockFromPC(self, block): # pylint: disable=invalid-name, unused-argument
        """
        callback function that Data fetching
        """
        sys.stdout.flush()
        return 0

    def ns_rpc_data_computeOnPC( # pylint: disable=invalid-name
            self,
            in_block,       # like a request block from EVB
            IsRecordBlock):  # send the result_block to EVB
        """
        callback function that sending result_block to EVB
            that indicating to record or stop
        """
        self.lock.acquire()
        is_record = self.is_record[0]
        self.lock.release()
        if (in_block.cmd == GenericDataOperations_EvbToPc.common.command.extract_cmd) and (
            in_block.description == "CalculateMFCC_Please"):

            data2pc = [is_record]
            IsRecordBlock.value = GenericDataOperations_EvbToPc.common.dataBlock(
                description ="*\0",
                dType       = GenericDataOperations_EvbToPc.common.dataType.uint8_e,
                cmd         = GenericDataOperations_EvbToPc.common.command.generic_cmd,
                buffer      = bytearray(data2pc),
                length      = len(data2pc),
            )
        sys.stdout.flush()
        return 0

class VisualDataClass:
    """
    Visual the audio data from EVB
    """
    def __init__(self, databuf, lock, is_record, event_stop, cyc_count):
        self.databuf = databuf
        self.lock    = lock
        self.is_record = is_record
        self.event_stop = event_stop
        self.cyc_count = cyc_count
        secs2show = FRAMES_TO_SHOW * HOP_SIZE/SAMPLING_RATE
        self.xdata = np.arange(FRAMES_TO_SHOW * HOP_SIZE) / SAMPLING_RATE
        self.fig, self.ax_handle = plt.subplots()
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        plt.subplots_adjust(bottom=0.35)
        self.lock.acquire()
        np_databuf = databuf[0:]
        self.lock.release()
        self.line_data, = self.ax_handle.plot(self.xdata, np_databuf, lw=0.5, color = 'blue')
        plt.ylim([-1.1,1.1])
        self.ax_handle.set_xlim((0, secs2show))
        self.ax_handle.set_xlabel('Time (Seconds)')
        plt.plot(
            [0, secs2show],
            [1, 1],
            color='black',
            lw=1)
        plt.plot(
            [0, secs2show],
            [-1, -1],
            color='black',
            lw=1)
        # making buttons
        def make_button(pos, name, callback_func):
            ax_button = plt.axes(pos)
            button = Button(
                        ax_button,
                        name,
                        color = 'w',
                        hovercolor = 'aliceblue')
            button.label.set_fontsize(16)
            button.on_clicked(callback_func)
            return button
        self.wavfile = None
        self.button_replay = make_button(
                            [0.35, 0.15, 0.14, 0.075],
                            'stop',
                            self.callback_recordstop)
        self.button_record = make_button(
                            [0.5, 0.15, 0.14, 0.075],
                            'record',
                            self.callback_recordstart)
        plt.show()

    def handle_close(self, event): # pylint: disable=unused-argument
        """
        Finish everything when you close your plot
        """
        self.lock.acquire()
        self.is_record[0] = 0
        self.lock.release()
        print('Window close')
        time.sleep(0.05)
        self.event_stop.set() # let main function know program should be terminated now

    def callback_recordstop(self, event):
        """
        for stop button
        """
        self.lock.acquire()
        self.is_record[0] = 0
        self.lock.release()
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def callback_recordstart(self, event):
        """
        for record button
        """
        self.lock.acquire()
        is_record = self.is_record[0]
        self.lock.release()
        if is_record == 0:
            self.lock.acquire()
            self.is_record[0] = 1
            self.lock.release()
            while 1:
                self.lock.acquire()
                cyc_count = self.cyc_count[0]
                np_databuf = self.databuf[0:]
                self.lock.release()
                zeros_tail = [0.0] * (HOP_SIZE * (FRAMES_TO_SHOW - cyc_count))
                np_databuf = np_databuf[:HOP_SIZE*cyc_count] + zeros_tail
                self.line_data.set_data(self.xdata, np_databuf)

                plt.pause(0.05)
                self.lock.acquire()
                is_record = self.is_record[0]
                self.lock.release()
                if is_record == 0:
                    break
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

def target_proc_draw(databuf, lock, recording, event_stop, cyc_count):
    """
    one of multiprocesses: draw
    """
    VisualDataClass(databuf, lock, recording, event_stop, cyc_count)

def target_proc_evb2pc(tty, baud, databuf, wavout, lock, is_record, cyc_count):
    """
    one of multiprocesses: EVB sends data to PC
    """
    transport_evb2pc = erpc.transport.SerialTransport(tty, int(baud))
    handler = DataServiceClass(databuf, wavout, lock, is_record, cyc_count)
    service = GenericDataOperations_EvbToPc.server.evb_to_pcService(handler)
    server = erpc.simple_server.SimpleServer(transport_evb2pc, erpc.basic_codec.BasicCodec)
    server.add_service(service)
    print("\r\nServer started - waiting for EVB to send an eRPC request")
    sys.stdout.flush()
    server.run()

def main(args):
    """
    main
    """
    event_stop = multiprocessing.Event()
    lock = Lock()
    databuf = Array('d', FRAMES_TO_SHOW * HOP_SIZE)
    record_ind  = Array('i', [0]) # is_record indicator. 'No record' as initialization
    cyc_count   = Array('i', [0])
    # we use two multiprocesses to handle real-time visualization and recording
    # 1. proc_draw   : to visualize
    # 2. proc_evb2pc : to capture data from evb and recording
    proc_draw   = Process(
                    target = target_proc_draw,
                    args   = (databuf,lock, record_ind, event_stop, cyc_count))
    proc_evb2pc = Process(
                    target = target_proc_evb2pc,
                    args   = (  args.tty,
                                args.baud,
                                databuf,
                                args.out,
                                lock,
                                record_ind,
                                cyc_count))
    proc_draw.start()
    proc_evb2pc.start()
    # monitor if program should be terminated
    while True:
        if event_stop.is_set():
            proc_draw.terminate()
            proc_evb2pc.terminate()
            #Terminating main process
            sys.exit(1)
        time.sleep(0.5)

if __name__ == "__main__":

    # parse cmd parameters
    argParser = argparse.ArgumentParser(description="NeuralSPOT GenericData RPC Demo")

    argParser.add_argument(
        "-w",
        "--tty",
        default = "/dev/tty.usbmodem1234561", # "/dev/tty.usbmodem1234561"
        help    = "Serial device (default value is None)",
    )
    argParser.add_argument(
        "-B",
        "--baud",
        default = "115200",
        help    = "Baud (default value is 115200)"
    )
    argParser.add_argument(
        "-o",
        "--out",
        default = "audio.wav",
        help    = "File where data will be written (default is audio.wav",
    )

    main(argParser.parse_args())

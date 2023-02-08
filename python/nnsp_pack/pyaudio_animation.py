"""
Module to record the speech and display it
"""
import os
import wave
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import pyaudio
plt.style.use('bmh')
LINE_MINMAX         = [-1.1, 1.1]
class AudioShowClass:
    """
    Class to record the speech and display it
    """
    def __init__(
            self,
            record_seconds          = 6,
            sample_rate             = 16000,
            frame_size              = 160,
            wave_output_filename    = "speech.wav",
            non_stop                = False,
            proc_st                 = None):

        self.record_seconds = record_seconds
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_time_record = float(frame_size) / float(sample_rate)
        self.frame_time_replay = 0.05
        self.wave_output_filename = wave_output_filename

        self.num_blks = int(self.sample_rate / self.frame_size * self.record_seconds)
        self.data_buffer= np.zeros(
            (self.sample_rate * self.record_seconds,),
            dtype = float)
        self.data_trigger= np.zeros(
            (self.sample_rate * self.record_seconds,),
            dtype = float)
        self.counts_frames = 0

        self.non_stop = non_stop
        self.const_data_buffer = np.arange(self.record_seconds * self.sample_rate) \
                                                / self.sample_rate
        self.proc_st = proc_st
        self.draw_lock = 0
        self.fig, ax_handle = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        ax_handle.set_xlim((0, self.record_seconds))
        ax_handle.set_ylim(LINE_MINMAX)
        plt.title("\"Record\" your voice and close the window to continue")
        self.line_data, = ax_handle.plot([], [], lw=0.2, color = 'blue')
        self.line_stop, = ax_handle.plot([], [], lw=0.2, color = 'k')
        ax_handle.set_xlabel('Time (Seconds)')
        plt.plot(
            self.const_data_buffer,
            self.const_data_buffer * 0 + 1,
            color='black',
            lw=1)
        plt.plot(
            self.const_data_buffer,
            self.const_data_buffer * 0 - 1,
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
                            'replay',
                            self.callback_replay)
        self.button_record = make_button(
                            [0.5, 0.15, 0.14, 0.075],
                            'record',
                            self.callback_record)
        if os.path.exists(self.wave_output_filename):
            self.data_buffer, _= sf.read(self.wave_output_filename)
            if len(self.data_buffer) != len(self.const_data_buffer):
                self.data_buffer = self.const_data_buffer * 0
            self.line_data.set_data(self.const_data_buffer, self.data_buffer)
        self.lock_button = 0
        plt.show()

    def callback_record(self, event):
        '''
        Callback for start recording
        '''
        if self.lock_button == 0:
            self.lock_button = 1
            plt.figure(self.fig)
            self.wavfile = wave.open(self.wave_output_filename, 'wb')
            self.wavfile.setnchannels(1)
            self.wavfile.setsampwidth(2)
            self.wavfile.setframerate(self.sample_rate)
            print("start recording...")
            audio_handle = pyaudio.PyAudio() # instantiate PyAudio
            self.draw_lock = 0
            self.counts_frames = 0

            def callback_streamin(in_data, frame_count, time_info, status):
                """
                Callback function for frame process
                """
                self.draw_lock = 1
                self.wavfile.writeframesraw(in_data)
                data_fr = np.frombuffer(in_data, dtype=np.int16) / 32768
                start = self.frame_size * self.counts_frames
                self.data_buffer[start:start+self.frame_size] = data_fr

                if self.proc_st:
                    trig = self.proc_st.frame_proc(data_fr)
                    self.data_trigger[start:start+self.frame_size]=trig

                if self.counts_frames == (self.num_blks-1):
                    if self.non_stop:
                        self.data_buffer    *= 0
                        self.data_trigger   *= 0
                        ret = (in_data, pyaudio.paContinue)
                    else:
                        ret = (in_data, pyaudio.paAbort)
                else:
                    ret = (in_data, pyaudio.paContinue)

                self.counts_frames = (self.counts_frames + 1) % self.num_blks
                self.draw_lock = 0
                return ret

            stream=audio_handle.open(
                    format              = pyaudio.paInt16,
                    channels            = 1,
                    rate                = self.sample_rate,
                    input               = True,
                    frames_per_buffer   = self.frame_size,
                    stream_callback     = callback_streamin)

            # stop and close the audio stream
            self.data_buffer *= 0
            self.line_data.set_data(self.const_data_buffer, self.data_buffer)
            stream.start_stream()
            while stream.is_active():
                if self.draw_lock == 0:
                    self.line_data.set_data(self.const_data_buffer, self.data_buffer)
                    ending = self.frame_time_record * self.counts_frames
                    self.line_stop.set_data([ending, ending], LINE_MINMAX)
                plt.pause(self.frame_time_replay)
            self.line_stop.set_data([0, 0], LINE_MINMAX)
            stream.stop_stream()
            stream.close()
            audio_handle.terminate()
            self.wavfile.close()
            print("stop recording")
            self.lock_button = 0
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def _playsound(self, wavname):
        """
        play wavename
        """
        event_obj = threading.Event()
        with sf.SoundFile(wavname) as wavefile:
            def callback_streamout(outdata, framesize, time, status):
                self.draw_lock  = 1
                data            = wavefile.buffer_read(framesize, dtype='float32')
                if len(outdata) > len(data):
                    outdata[:len(data)] = data
                    outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
                    raise sd.CallbackStop
                outdata[:] = data
                self.counts_frames += 1
                self.draw_lock = 0

            self.counts_frames = 0
            self.data_buffer *= 0
            stream = sd.RawOutputStream(
                samplerate          = wavefile.samplerate,
                channels            = wavefile.channels,
                callback            = callback_streamout,
                blocksize           = self.frame_size,
                finished_callback   = event_obj.set)

            with stream: # with stream will do stream.start and stream.stop()
                while not event_obj.is_set():
                    if self.draw_lock == 0:
                        ending = self.frame_time_record * (self.counts_frames + 1)
                        self.line_stop.set_data(
                            [ending, ending],
                        LINE_MINMAX)
                    plt.pause(self.frame_time_replay)

                self.line_stop.set_data(
                            [0, 0],
                        LINE_MINMAX)

    def callback_replay(self, event):
        """
        replay the recording utterance
        """
        if self.lock_button == 0:
            self.lock_button = 1
            if os.path.exists(self.wave_output_filename):
                print('start playing')
                self._playsound(self.wave_output_filename)
                print('end playing')
            self.lock_button = 0

        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

if __name__ == "__main__":
    aud_handle = AudioShowClass(record_seconds=6, non_stop=False)

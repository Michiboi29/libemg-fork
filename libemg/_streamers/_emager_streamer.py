import time
import serial # pyserial
import serial.tools.list_ports
import numpy as np
from multiprocessing import Process
from libemg.shared_memory_manager import SharedMemoryManager

def reorder(data, mask, match_result, packet_size=128):
    '''
    Use to find the start of a packet in the data array and reorder it from the found starting point.
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    number_of_packet = int(len(data)//packet_size)
    roll_data = []
    for i in range(number_of_packet):
        data_lsb = data[i*packet_size:(i+1)*packet_size] & np.ones(packet_size, dtype=np.int8)
        mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
        try:
            offset = np.where(mask_match == match_result)[0][0] - 3
        except IndexError:
            print("No match found")
            return None
        roll_data.append(np.roll(data[i*packet_size:(i+1)*packet_size], -offset))
    return roll_data

class Emager:
    def __init__(self, baud_rate, specified_port=None):
        if specified_port:
            com_port = specified_port
        else:
            # find the port
            vid = 0x04b4
            pid = 0xf155
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if p.vid == vid and p.pid == pid:
                    com_port = p.device
                    break

        try:
            self.ser = serial.Serial(com_port, baud_rate, timeout=1)
            print("Serial port connected successfully. " + com_port + " at " + str(baud_rate) + " baud rate.")
        except serial.SerialException as e:
            print("Failed to connect to serial port: " + com_port, str(e))
            return
        
        self.num_channels = 64
        self.packet_size = 128
        ### ^ Number of bytes in message (i.e. channel bytes + header/tail bytes)
        self.mask = np.array([0, 2] + [0, 1] * (self.num_channels-1))
        ### ^ Template mask for template matching on input data
        self.channelMap = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
                          [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
                          [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
                          [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
        self.emg_handlers = []

    def connect(self):
        if self.ser is not None:
            if not self.ser.is_open:
                self.ser.open()
        return

    def add_emg_handler(self, closure):
        self.emg_handlers.append(closure)

    def run(self):
        self.connect()
        samples = np.zeros(self.num_channels)
        while True:
            # wait for data
            bytesToRead = 0
            while bytesToRead < self.packet_size:
                bytes_available = self.ser.in_waiting
                bytesToRead = bytes_available - (bytes_available % self.packet_size)
                time.sleep(0.02)

            # read data if there was data
            raw_data_packet = self.ser.read(bytesToRead)
            # find start of packet
            data_packet = reorder(list(raw_data_packet), self.mask, self.num_channels-1, self.packet_size)
            if data_packet is None:
                print("No valid packets found")
                continue
            if len(data_packet) > 0:
                for p in range(len(data_packet)):
                    # Fuse two bytes into one 16-bit integer
                    samples = [int.from_bytes(bytes([data_packet[p][s*2], data_packet[p][s*2+1]]), 'big',signed=True) for s in range(self.num_channels)]

                    # Reorder the samples according to the channel map
                    samples = np.array(samples)[self.channelMap]
                    # Add the samples to the shared memory
                    for h in self.emg_handlers:
                        h(samples)
    
    def clear_buffer(self):
        '''
        Clear the serial port input buffer.
        :return: None
        '''
        if self.ser is not None:
            self.ser.reset_input_buffer()
        return

    def close(self):
        if self.ser is not None:
            if self.ser.is_open:
                self.ser.close()
        return

# Myostreamer begins here ------
class EmagerStreamer(Process):
    def __init__(self, shared_memory_items, specified_port=None):
        super().__init__(daemon=True)
        self.smm = SharedMemoryManager()
        self.shared_memory_items = shared_memory_items
        self.specified_port = specified_port

    def run(self):
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        
        e = Emager(1500000, self.specified_port)
        e.connect()

        def write_emg(emg):
            emg = np.array(emg)
            self.smm.modify_variable('emg', lambda x: np.vstack((emg, x))[:x.shape[0], :])
            self.smm.modify_variable('emg_count', lambda x: x + 1)
            
        e.add_emg_handler(write_emg)

        while True:
            try:
                e.run()
            except Exception as exception:
                print("Error Occured. " + str(exception))
                quit() 


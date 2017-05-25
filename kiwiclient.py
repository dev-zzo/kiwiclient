#!/usr/bin/env python

import array
import logging
import socket
import struct
import time

import wsclient

#
# IMAADPCM decoder
#

stepSizeTable = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34,
    37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
    544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
    1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026,
    4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767)

indexAdjustTable = [
    -1, -1, -1, -1,  # +0 - +3, decrease the step size
     2, 4, 6, 8,     # +4 - +7, increase the step size
    -1, -1, -1, -1,  # -0 - -3, decrease the step size
     2, 4, 6, 8      # -4 - -7, increase the step size
]


def clamp(x, xmin, xmax):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x

class ImaAdpcmDecoder(object):
    def __init__(self):
        self.index = 0
        self.prev = 0

    def _decode_sample(self, code):
        step = stepSizeTable[self.index]
        self.index = clamp(self.index + indexAdjustTable[code], 0, len(stepSizeTable) - 1)
        difference = step >> 3
        if ( code & 1 ):
            difference += step >> 2
        if ( code & 2 ):
            difference += step >> 1
        if ( code & 4 ):
            difference += step
        if ( code & 8 ):
            difference = -difference
        sample = clamp(self.prev + difference, -32768, 32767)
        self.prev = sample
        return sample

    def decode(self, data):
        samples = array.array('h')
        if isinstance(data, str):
            for b in data:
                b = ord(b)
                sample0 = self._decode_sample(b & 0x0F)
                sample1 = self._decode_sample(b >> 4)
                samples.append(sample0)
                samples.append(sample1)
        else:
            for b in data:
                sample0 = self._decode_sample(b & 0x0F)
                sample1 = self._decode_sample(b >> 4)
                samples.append(sample0)
                samples.append(sample1)
        return samples


#
# KiwiSDR WebSocket client
#

class KiwiError(Exception):
    pass
class KiwiTooBusyError(KiwiError):
    pass
class KiwiDownError(KiwiError):
    pass
class KiwiBadPasswordError(KiwiError):
    pass

class KiwiSDRClientBase(object):
    """KiwiSDR WebSocket client."""

    def __init__(self):
        self._socket = None
        self._sample_rate = None

    def connect(self, host, port):
        self._socket = socket.socket()
        self._socket.settimeout(self._options.socket_timeout)
        self._socket.connect((host, port))
        self._prepare_stream(host, port, '/%d/SND' % int(time.time()))

    def _prepare_stream(self, host, port, which):
        import mod_pywebsocket.common
        from mod_pywebsocket.stream import Stream
        from mod_pywebsocket.stream import StreamOptions

        handshake = wsclient.ClientHandshakeProcessor(self._socket, host, port)
        handshake.handshake(which)

        request = wsclient.ClientRequest(self._socket)
        request.ws_version = mod_pywebsocket.common.VERSION_HYBI13

        stream_option = StreamOptions()
        stream_option.mask_send = True
        stream_option.unmask_receive = False

        self._stream = Stream(request, stream_option)

    def set_mod(self, mod, lc, hc, freq):
        self._stream.send_message('SET mod=%s low_cut=%d high_cut=%d freq=%.3f' % (mod, lc, hc, freq))

    def set_agc(self, on=False, hang=False, thresh=-100, slope=6, decay=1000, gain=50):
        self._stream.send_message('SET agc=%d hang=%d thresh=%d slope=%d decay=%d manGain=%d' % (on, hang, thresh, slope, decay, gain))

    def set_squelch(self, sq, thresh):
        self._stream.send_message('SET squelch=%d max=%d' % (sq, thresh))

    def set_autonotch(self, val):
        self._stream.send_message('SET autonotch=%d' % (val))

    def set_name(self, name):
        self._stream.send_message('SET name=%d' % (name))

    def _set_auth(self, client_type, password=''):
        self._stream.send_message('SET auth t=%s p=%s' % (client_type, password))

    def _set_ar_ok(self, ar_in, ar_out):
        self._stream.send_message('SET AR OK in=%d out=%d' % (ar_in, ar_out))

    def _set_gen(self, freq, attn):
        self._stream.send_message('SET genattn=%d' % (attn))
        self._stream.send_message('SET gen=%d mix=%d' % (freq, -1))

    def _set_keepalive(self):
        self._stream.send_message('SET keepalive')

    def _process_msg_param(self, name, value):
        print "%s: %s" % (name, value)
        if name == 'too_busy':
            raise KiwiTooBusyError('all %s client slots taken' % value)
        if name == 'badp' and value == '1':
            raise KiwiBadPasswordError()
        if name == 'down':
            raise KiwiDownError('server is down atm')
        if name == 'audio_rate':
            self._set_ar_ok(int(value), 44100)
        elif name == 'sample_rate':
            self._sample_rate = float(value)
            # Optional, but is it?..
            self.set_squelch(0, 0)
            self.set_autonotch(0)
            self._set_gen(0, 0)
            # Required to get rolling
            self._setup_rx_params()
            # Also send a keepalive
            self._set_keepalive()

    def _process_msg(self, body):
        for pair in body.split(' '):
            name, value = pair.split('=', 1)
            self._process_msg_param(name, value)

    def _process_aud(self, body):
        seq = struct.unpack('<I', body[:4])[0]
        smeter = struct.unpack('>H', body[4:6])[0]
        data = body[6:]
        rssi = (smeter & 0x0FFF) // 10 - 127
        self._process_samples(seq, self._decoder.decode(data), rssi)

    def _process_samples(self, seq, samples, rssi):
        pass

    def _setup_rx_params(self):
        self._set_mod('am', 100, 2800, 4625.0)
        self._set_agc(True)

    def run(self):
        """Run the client."""

        self._decoder = ImaAdpcmDecoder()
        try:
            self._set_auth('kiwi', '')
            # Loop forever
            while True:
                try:
                    received = self._stream.receive_message()
                except KeyboardInterrupt:
                    break
                # print repr(received)
                id, body = received.split(' ', 1)
                if id == 'MSG':
                    self._process_msg(body)
                elif id == 'SND':
                    self._process_aud(body)
                    # Ensure we don't get kicked due to timeouts
                    self._set_keepalive()
                else:
                    print 'Unknown message ID: %s' % (id)
                    print repr(received)
            self._stream.close_connection()
        finally:
            self._socket.close()

# EOF
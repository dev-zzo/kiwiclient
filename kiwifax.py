import array
import codecs
import logging
import os
import struct
import sys
import time
import png
import math
import cmath
import traceback
from optparse import OptionParser

import kiwiclient

def clamp(x, xmin, xmax):
    if x < xmin:
        x = xmin
    if x > xmax:
        x = xmax
    return x

def norm_clamp(x, xmin, xmax):
    return (clamp(x, xmin, xmax) - xmin) / (xmax - xmin)

def real2complex(x):
    return [ complex(x[i+0]-x[i+2], x[i+1]-x[i+3]) for i in xrange(0, len(x), 4) ]

def fm_detect(X, prev, angle):
    angle_coeff = cmath.rect(1, angle)
    vals = array.array('f')
    for x in X:
        vals.append(1 - cmath.phase(x * prev.conjugate() * angle_coeff) / math.pi)
        prev = x
    return vals

def dft_complex(input):
    width = len(input)
    output = []
    w1d = complex(0, -2 * math.pi / width)
    w1 = 0
    for k in xrange(width):
        X = 0
        w2d = cmath.exp(w1)
        w2 = complex(1, 0)
        for n in xrange(width):
            X += input[n] * w2
            w2 *= w2d
        output.append(X)
        w1 += w1d
    return output

def power_db(input):
    return [ 10 * math.log10(abs(x) / len(input)) for x in input ]

def dump_to_csv(filename, data):
    with open(filename, 'a') as fp:
        for x in data:
            fp.write("%.6f," % x)
        fp.write("\n")

def popcount_thresh(X, thresh):
    count = 0
    for x in X:
        if x:
            count += 1
    return count >= thresh

def interp_cubic(t, p0, p1, p2, p3):
    a0 = p3 - p2 - p0 + p1
    a1 = p0 - p1 - a0
    a2 = p2 - p0
    a3 = p1
    return a0 * (t*t*t) + a1 * (t*t) + a2 * t + a3

def interp_hermite(t, p0, p1, p2, p3):
    c0 = p1
    c1 = 0.5 * (p2 - p0)
    c2 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2)
    return c0 + (t * (c1 + (t * (c2 + (t * c3)))))

class Interpolator:
    def __init__(self, factor):
        self._buffer = array.array('f')
        self._t = 0
        self.set_factor(factor)
    def set_factor(self, factor):
        self._dt = factor
    def refill(self, samples):
        for x in samples:
            self._buffer.append(float(x))
    def _flush(self):
        t_int = math.trunc(self._t)
        t_new = min(t_int, len(self._buffer))
        # print "flush", t_int, t_new
        if t_new > 0:
            self._t -= t_new
            self._buffer = self._buffer[t_new:]
    def __iter__(self):
        return self
    def next(self):
        t_int = math.trunc(self._t)
        t_frac = self._t - t_int
        # print "pop", t_int, t_frac, len(self._buffer)
        if t_int + 3 >= len(self._buffer):
            self._flush()
            raise StopIteration()
        self._t += self._dt
        return interp_hermite(t_frac, self._buffer[t_int], self._buffer[t_int + 1], self._buffer[t_int + 2], self._buffer[t_int + 3])

def test_interp():
    I = Interpolator(0.4)
    I.refill([0,1,2,3,4,5])
    for x in I:
        print x
    I.refill([6,7,8])
    for x in I:
        print x

def peak_around(P, bin, delta):
    return sorted(P[bin-delta:bin+delta+1])[-1]

def mapper_df_to_intensity(dfs, black_thresh, white_thresh):
    for x in dfs:
        yield norm_clamp(x, black_thresh, white_thresh)

class KiwiFax(kiwiclient.KiwiSDRClientBase):
    def __init__(self, options):
        super(KiwiFax, self).__init__()
        self._options = options
        self._ioc = options.ioc
        self._lpm = options.lpm
        self._prevX = complex(0)
        
        self._resampler = Interpolator(1.0)
        
        self._startstop_buffer = []
        self._start_samples = [ False for x in xrange(16) ]
        self._stop_samples = [ False for x in xrange(16) ]
        self._startstop_index = 0

        self._state = 'idle'
        
        self._phasing_count = 0
        
        self._rows = []
        self._pixel_buffer = array.array('f')
        self._max_height = 99999
        self._new_roll()
        if options.force:
            self._switch_state('printing')

    def _switch_state(self, new_state):
        print "\nSwitching to: %s" % new_state
        self._state = new_state
        if new_state == 'starting':
            self._new_roll()
        elif new_state == 'phasing':
            self._phasing_count = 0

    def _setup_rx_params(self):
        self.set_mod('usb', 300, 2500, self._options.frequency)
        self.set_agc(True)

    def _process_samples(self, seq, samples, rssi):
        sys.stdout.write('\rBlock: %08x, RSSI: %04d %s' % (seq, rssi, self._state))
        samples = [ x / 32768.0 for x in samples ]

        X = real2complex(samples)
        sample_rate = self._sample_rate / 4
        self._process_startstop(X, sample_rate)
        self._process_pixels(X, sample_rate)

    def _process_startstop(self, samples, sample_rate):
        window_size = 512
        self._startstop_buffer.extend(samples)
        while len(self._startstop_buffer) >= window_size:
            window = self._startstop_buffer[:window_size]
            self._startstop_buffer = self._startstop_buffer[window_size:]
            P = power_db(dft_complex(window))
            #if self._state != 'idle':
            #    dump_to_csv(self._output_name + '-ss.csv', P)
            if True:
                nf_level = sorted(P)[len(P) // 2]
                # TODO: avoid hardcoded bins
                white_bin = 22
                black_bin = 193
                startstop_center_bin = 107
                detect_white = peak_around(P, white_bin, 5) - nf_level >= 10
                detect_black = peak_around(P, black_bin, 10) - nf_level >= 10
                if not detect_white and peak_around(P, startstop_center_bin, 10) - nf_level >= 10:
                    start_peak = max(peak_around(P, startstop_center_bin-64, 10), peak_around(P, startstop_center_bin+64, 10))
                    stop_peak = max(peak_around(P, startstop_center_bin-96, 10), peak_around(P, startstop_center_bin+96, 10))
                    detect_start = start_peak > stop_peak
                    detect_stop = stop_peak > start_peak
                else:
                    detect_start = detect_stop = False
                sys.stdout.write(" %.2f %s%s%s%s" % (nf_level, "wW"[detect_white], "bB"[detect_black], "sS"[detect_start], "tT"[detect_stop]))
                
                self._start_samples[self._startstop_index] = detect_start
                self._stop_samples[self._startstop_index] = detect_stop
                self._startstop_index += 1
                if self._startstop_index >= len(self._start_samples):
                    self._startstop_index = 0
                start_detected = popcount_thresh(self._start_samples, len(self._start_samples) * 3 / 4)
                stop_detected = popcount_thresh(self._stop_samples, len(self._stop_samples) * 3 / 4)

                if start_detected:
                    if self._state != 'starting':
                        print "\n\nSTART DETECTED\n"
                        if self._state == 'printing':
                            self._flush_rows()
                        self._switch_state('starting')
                else:
                    if self._state == 'starting':
                        self._switch_state('phasing')
                if stop_detected and self._state == 'printing':
                    print "\n\nSTOP DETECTED\n"
                    self._flush_rows()
                    self._switch_state('idle')

    def _new_roll(self):
        self._rows = []
        ts = time.strftime('%Y%m%dT%H%MZ', time.gmtime())
        self._output_name = '%s_%d' % (ts, int(self._options.frequency * 1000))

    def _process_pixels(self, samples, sample_rate):
        if not self._state in ('phasing', 'printing'):
            return
        detected = fm_detect(samples, self._prevX, -0.1 * math.pi)
        self._prevX = samples[-1]
        #dump_to_csv(self._output_name + '-discr.csv', detected)
        # Remap the detected region into [0,1)
        black_thresh, white_thresh = 0.45, 1.0
        pixels = array.array('f', mapper_df_to_intensity(detected, black_thresh, white_thresh))
        # Scale and adjust pixel rate
        samples_per_line = sample_rate * 60.0 / self._lpm
        # Sane aspect ration comes with width = 192000/lpm (px)
        pixels_per_line = int((1600 * 120) / self._lpm)
        self._resampler.set_factor(samples_per_line / pixels_per_line)
        self._resampler.refill(pixels)
        self._pixel_buffer.extend(self._resampler)
        
        if self._state == 'phasing':
            # Count attempts at phasing to avoid getting stuck
            self._phasing_count += 1
            # TODO: skip 3-4 lines
            if self._phasing_count <= 20:
                phasing_pulse_size = 70
                i = 0
                while i + phasing_pulse_size < len(self._pixel_buffer):
                    s = 0
                    for j in xrange(i, i + phasing_pulse_size):
                        s += clamp(self._pixel_buffer[j], 0, 1)
                    s /= phasing_pulse_size
                    if s >= 0.8:
                        self._pixel_buffer = self._pixel_buffer[i + phasing_pulse_size * 3 // 4:]
                        print "Phasing OK"
                        self._switch_state('printing')
                        break
                    i += 1
                else:
                    self._pixel_buffer = self._pixel_buffer[min(0, i - phasing_pulse_size):]
            else:
                print "Phasing failed!"
                self._switch_state('printing')
        else:
            # Cut into rows of pixels
            while len(self._pixel_buffer) >= pixels_per_line:
                row = self._pixel_buffer[:pixels_per_line]
                new_buffer = self._pixel_buffer[pixels_per_line:]
                self._pixel_buffer = new_buffer
                self._process_row(row)

    def _process_row(self, row):
        pixels = array.array('B')
        for x in row:
            pixels.append(int(clamp(x, 0, 1) * 255))
        self._rows.append(pixels)
        if len(self._rows) >= self._max_height:
            print "Length exceeded; cutting the paper"
            self._new_roll()
        if len(self._rows) % 16:
            return
        self._flush_rows()

    def _flush_rows(self):
        if not self._rows:
            return
        with open(self._output_name + '.png', 'wb') as fp:
            while True:
                try:
                    png.Writer(len(self._rows[0]), len(self._rows), greyscale=True).write(fp, self._rows)
                    break
                except KeyboardInterrupt:
                    pass

def main():
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    parser = OptionParser()
    parser.add_option('--log-level', '--log_level', type='choice',
                      dest='log_level', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level.')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout', type='int', default=10,
                      help='Timeout(sec) for sockets')
    parser.add_option('-s', '--server-host', '--server_host',
                      dest='server_host', type='string',
                      default='localhost', help='server host')
    parser.add_option('-p', '--server-port', '--server_port',
                      dest='server_port', type='int',
                      default=8073, help='server port')

    parser.add_option('-f', '--freq',
                      dest='frequency',
                      type='float',
                      help='Frequency to tune to, in kHz.')
    parser.add_option('--station', '--station',
                      dest='station',
                      type='string', default=None,
                      help='Station ID to be appended')
    parser.add_option('-F', '--force-start',
                      dest='force',
                      action='store_true', default=False,
                      help='Frequency to tune to, in kHz.')
    parser.add_option('-i', '--ioc',
                      dest='ioc',
                      type='int', default=576,
                      help='Index of cooperation.')
    parser.add_option('-l', '--lpm',
                      dest='lpm',
                      type='int', default=120,
                      help='Lines per minute.')
    parser.add_option('--sr-coeff', '--sr-coeff',
                      dest='sr_coeff',
                      type='float', default=1.0,
                      help='Sample frequency correction; increase to make lines shorter, decrease to make lines longer')

    (options, unused_args) = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()))

    while True:
        recorder = KiwiFax(options)
        
        # Connect
        try:
            recorder.connect(options.server_host, options.server_port)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()
            print "Failed to connect, sleeping and reconnecting"
            time.sleep(15)
            continue
        # Record
        try:
            recorder.run()
            break
        except (kiwiclient.KiwiTooBusyError, kiwiclient.KiwiBadPasswordError):
            print "Server too busy now"
            time.sleep(15)
            continue
        except Exception as e:
            traceback.print_exc()
            break


if __name__ == '__main__':
    #test_interp()
    #sys.exit(0)
    main()
# EOF
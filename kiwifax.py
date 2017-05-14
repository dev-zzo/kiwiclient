import array
import codecs
import logging
import os
import struct
import sys
import time
import math
import cmath
import traceback
from optparse import OptionParser

import kiwiclient
import png


# Known bugs:
# * Start/stop detection doesn't work for 60 RPM


def dump_to_csv(filename, data):
    with open(filename, 'a') as fp:
        for x in data:
            fp.write("%.6f," % x)
        fp.write("\n")


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

def popcount_thresh(X, thresh):
    count = 0
    for x in X:
        if x:
            count += 1
    return count >= thresh

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

        self._state = 'idle'

        self._startstop_buffer = []
        self._startstop_score = 0

        self._prevX = complex(0)
        self._phasing_count = 0
        self._resampler = Interpolator(1.0)
        self._sf = 1.0 - 1e-6 * options.sr_coeff
        self._rows = []
        self._pixel_buffer = array.array('f')
        self._pixels_per_line = 1809
        self._max_height = 4000

        self._new_roll()
        if options.force:
            self._switch_state('printing')

    def _switch_state(self, new_state):
        logging.info("Switching to: %s", new_state)
        self._state = new_state
        if new_state == 'idle':
            self._startstop_score = 0
        elif new_state == 'starting':
            pass
        elif new_state == 'phasing':
            self._new_roll()
            self._phasing_count = 0
        elif new_state == 'printing':
            self._startstop_score = 0

    def _setup_rx_params(self):
        #self.set_mod('usb', 300, 2500, self._options.frequency - 1.9)
        self.set_mod('usb', 1500-1000, 2300+1000, self._options.frequency - 1.9)
        self.set_agc(True)

    def _process_samples(self, seq, samples, rssi):
        logging.info('Block: %08x, RSSI: %04d %s', seq, rssi, self._state)
        samples = [ x / 32768.0 for x in samples ]

        X = real2complex(samples)
        sample_rate = self._sample_rate / 4
        self._process_startstop(X, sample_rate)
        self._process_pixels(X, sample_rate)

    def _startstop_adjust(self, updown):
        if updown:
            self._startstop_score += 1
        else:
            self._startstop_score -= 2
            if self._startstop_score < 0:
                self._startstop_score = 0

    def _process_startstop(self, samples, sample_rate):
        window_size = 512
        self._startstop_buffer.extend(samples)
        while len(self._startstop_buffer) >= window_size:
            window = self._startstop_buffer[:window_size]
            self._startstop_buffer = self._startstop_buffer[window_size:]
            P = power_db(dft_complex(window))
            if self._options.dump_spectra and self._state != 'idle':
                dump_to_csv(self._output_name + '-ss.csv', P)
            # 5dB is added due to empiric observations of the target peak
            # values being at about +5dB ref the measured noise floor
            nf_level = sorted(P)[len(P) // 2] + 5
            # TODO: avoid hardcoded bins
            white_bin = 22
            black_bin = 193
            startstop_center_bin = 107
            detect_white = peak_around(P, white_bin, 10) - nf_level >= 10
            detect_black = peak_around(P, black_bin, 10) - nf_level >= 10
            startstop_peak = peak_around(P, startstop_center_bin, 10) - nf_level
            startstop_thresh = 7
            detect_startstop = startstop_peak >= startstop_thresh
            startstop_peak2 = 0

            if self._state in ('idle', 'starting'):
                startstop_peak2 = max(peak_around(P, startstop_center_bin-64, 10), peak_around(P, startstop_center_bin+64, 10)) - nf_level
                self._startstop_adjust(detect_startstop and startstop_peak2 >= startstop_thresh)
                if self._state == 'idle':
                    if self._startstop_score > 10:
                        logging.critical("START DETECTED")
                        self._switch_state('starting')
                else:
                    if self._startstop_score < 3:
                        self._switch_state('phasing')

            elif self._state == 'printing':
                startstop_peak2 = max(peak_around(P, startstop_center_bin-96, 10), peak_around(P, startstop_center_bin+96, 10)) - nf_level
                self._startstop_adjust(detect_startstop and startstop_peak2 >= startstop_thresh)
                if self._startstop_score > 10:
                    logging.critical("STOP DETECTED")
                    self._flush_rows()
                    self._switch_state('idle')

            logging.info("NF=%06.2f X1=%+06.2f X2=%+06.2f SS=%02d %s%s%s",
                nf_level,
                startstop_peak, startstop_peak2, self._startstop_score,
                "wW"[detect_white], "bB"[detect_black], "xX"[detect_startstop])

    def _new_roll(self):
        self._rows = []
        ts = time.strftime('%Y%m%dT%H%MZ', time.gmtime())
        self._output_name = '%s_%d' % (ts, int(self._options.frequency * 1000))
        if self._options.station:
            self._output_name += '_' + self._options.station

    def _process_pixels(self, samples, sample_rate):
        if not self._state in ('phasing', 'printing'):
            return
        detected = fm_detect(samples, self._prevX, -0.1 * math.pi)
        self._prevX = samples[-1]
        #dump_to_csv(self._output_name + '-discr.csv', detected)
        # Remap the detected region into [0,1)
        black_thresh, white_thresh = 0.425, 0.95
        pixels = array.array('f', mapper_df_to_intensity(detected, black_thresh, white_thresh))
        # Scale and adjust pixel rate
        samples_per_line = sample_rate * 60.0 / self._lpm
        resample_factor = (samples_per_line / self._pixels_per_line) * self._sf
        self._resampler.set_factor(resample_factor)
        self._resampler.refill(pixels)
        self._pixel_buffer.extend(self._resampler)

        if self._state == 'phasing':
            self._process_phasing()
        else:
            # Cut into rows of pixels
            while len(self._pixel_buffer) >= self._pixels_per_line:
                row = self._pixel_buffer[:self._pixels_per_line]
                new_buffer = self._pixel_buffer[self._pixels_per_line:]
                self._pixel_buffer = new_buffer
                self._process_row(row)

    def _process_phasing(self):
        # Count attempts at phasing to avoid getting stuck
        self._phasing_count += 1
        # Skip 3-4 lines; it seems phasing is not reliable when started right away
        if self._phasing_count <= 3:
            self._pixel_buffer = self._pixel_buffer[self._pixels_per_line:]
            return
        if self._phasing_count <= 100:
            phasing_pulse_size = 70
            i = 0
            while i + phasing_pulse_size < len(self._pixel_buffer):
                s = 0
                for j in xrange(i, i + phasing_pulse_size):
                    s += clamp(self._pixel_buffer[j], 0, 1)
                s /= phasing_pulse_size
                if s >= 0.8:
                    self._pixel_buffer = self._pixel_buffer[i + phasing_pulse_size * 3 // 4:]
                    logging.info("Phasing OK")
                    self._switch_state('printing')
                    break
                i += 1
            else:
                self._pixel_buffer = self._pixel_buffer[max(0, i - phasing_pulse_size):]
            return
        logging.error("Phasing failed! Starting anyway")
        self._switch_state('printing')

    def _process_row(self, row):
        pixels = array.array('B')
        for x in row:
            pixels.append(int(clamp(x, 0, 1) * 255))
        self._rows.append(pixels)
        if len(self._rows) % 16:
            return
        self._flush_rows()
        if len(self._rows) >= self._max_height:
            logging.info("Length exceeded; cutting the paper")
            self._new_roll()

    def _flush_rows(self):
        if not self._rows:
            return
        while True:
            with open(self._output_name + '.png', 'wb') as fp:
                try:
                    png.Writer(len(self._rows[0]), len(self._rows), greyscale=True).write(fp, self._rows)
                    break
                except KeyboardInterrupt:
                    pass

def main():
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    parser = OptionParser()
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
                      help='Frequency to tune to, in kHz (will be tuned down by 1.9kHz)')
    parser.add_option('--station', '--station',
                      dest='station',
                      type='string', default=None,
                      help='Station ID to be appended to file names')
    parser.add_option('-F', '--force-start',
                      dest='force',
                      action='store_true', default=False,
                      help='Force the decoding without waiting for start tone or phasing')
    parser.add_option('-i', '--ioc',
                      dest='ioc',
                      type='int', default=576,
                      help='Index of cooperation; default: 576.')
    parser.add_option('-l', '--lpm',
                      dest='lpm',
                      type='int', default=120,
                      help='Lines per minute; default: 120.')
    parser.add_option('--sr-coeff', '--sr-coeff',
                      dest='sr_coeff',
                      type='float', default=0,
                      help='Sample frequency correction, ppm; positive if the lines are too short; negative otherwise')
    parser.add_option('--dump-spectra', '--dump-spectra',
                      dest='dump_spectra',
                      action='store_true', default=False,
                      help='Dump block spectra to a CSV file')

    (options, unused_args) = parser.parse_args()

    # Setup logging
    fmtr = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y%m%dT%H%MZ')
    fmtr.converter = time.gmtime
    fh = logging.FileHandler('log_%s_%d_%d.log' % (options.server_host, options.server_port, int(options.frequency * 1000)))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmtr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmtr)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    rootLogger.addHandler(fh)
    rootLogger.addHandler(ch)

    logging.critical('* * * * * * * *')
    logging.critical('Logging started')
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
            print "Server too busy now, sleeping and reconnecting"
            time.sleep(15)
            continue
        except Exception as e:
            traceback.print_exc()
            break


if __name__ == '__main__':
    main()
# EOF
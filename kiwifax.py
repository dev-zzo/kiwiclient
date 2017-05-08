import array
import codecs
import logging
import os
import struct
import sys
import time
import png
import dsp
import math
import cmath
import traceback
from optparse import OptionParser

import kiwiclient

def norm_clamp(x, xmin, xmax):
    if x < xmin:
        x = xmin
    if x > xmax:
        x = xmax
    return (x - xmin) / (xmax - xmin)

def real2complex(x):
    return [ complex(x[i+0]-x[i+2], x[i+1]-x[i+3]) * (1/32768.0) for i in xrange(0, len(x), 4) ]

def fm_detect(X, prev):
    vals = []
    for x in X:
        vals.append(1 - cmath.phase(x * prev.conjugate()) / math.pi)
        prev = x
    return vals

class KiwiFax(kiwiclient.KiwiSDRClientBase):
    def __init__(self, options):
        super(KiwiFax, self).__init__()
        self._options = options
        self._start_ts = None
        self._last_seq = 0
        self._prevX = complex(0)
        self._ioc = 576
        #self._lpm = 60
        self._lpm = 120
        self._rows = []
        self._current_row = array.array('B')

    def _setup_rx_params(self):
        mod = self._options.modulation
        lp_cut = self._options.lp_cut
        hp_cut = self._options.hp_cut
        freq = self._options.frequency
        if (mod == 'am'):
            # For AM, ignore the low pass filter cutoff
            lp_cut = -hp_cut
        self.set_mod(mod, lp_cut, hp_cut, freq)
        self.set_agc(True)

    def _process_samples(self, seq, samples, rssi):
        sys.stdout.write('\rBlock: %08x, RSSI: %-04d' % (seq, rssi))
        if seq - 1 != self._last_seq:
            print " SEQ CHACK FAIL -- packet loss?!"
        self._last_seq = seq
        self._decode(samples)

    def _get_output_filename(self):
        ts = time.strftime('%Y%m%dT%H%MZ', self._start_ts)
        sta = '' if self._options.station is None else '_' + self._options.station
        return '%s_%d_%s%s' % (ts, int(self._options.frequency * 1000), self._options.modulation, sta)

    def _decode(self, samples):
        X = real2complex(samples)
        ph = fm_detect(X, self._prevX)
        self._prevX = X[-1]
        for x in ph:
            x = math.trunc(norm_clamp(x, 0.25, 0.85) * 255)
            self._current_row.append(x)
        samples_per_line = int((self._sample_rate / 4) * 60 / self._lpm)
        while len(self._current_row) >= samples_per_line:
            self._rows.append(self._current_row[:samples_per_line])
            self._current_row = self._current_row[samples_per_line:]
        if (len(self._rows) % 16) == 15:
            with open('data.png', 'wb') as fp:
                while True:
                    try:
                        png.Writer(samples_per_line, len(self._rows), greyscale=True).write(fp, self._rows)
                        break
                    except KeyboardInterrupt:
                        pass
        
        #with open('data.csv', 'a') as fp:
        #    for x in ph:
        #        fp.write("%.6f\n" % x)

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
    parser.add_option('-m', '--modulation',
                      dest='modulation',
                      type='string', default='am',
                      help='Modulation; one of am, lsb, usb, cw, nbfm')
    parser.add_option('-L', '--lp-cutoff',
                      dest='lp_cut',
                      type='float', default=100,
                      help='Low-pass cutoff frequency, in Hz.')
    parser.add_option('-H', '--hp-cutoff',
                      dest='hp_cut',
                      type='float', default=2600,
                      help='Low-pass cutoff frequency, in Hz.')
    parser.add_option('--station', '--station',
                      dest='station',
                      type='string', default=None,
                      help='Station ID to be appended')

    (options, unused_args) = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()))

    while True:
        recorder = KiwiFax(options)
        
        # Connect
        try:
            recorder.connect(options.server_host, options.server_port)
        except KeyboardInterrupt:
            break
        except:
            print "Failed to connect, sleeping and reconnecting"
            time.sleep(15)
            continue
        # Record
        try:
            recorder.run()
            break
        except KeyboardInterrupt:
            break
        except kiwiclient.KiwiTooBusyError:
            print "Server too busy now"
            time.sleep(15)
            continue
        except Exception as e:
            traceback.print_exc()
            break


if __name__ == '__main__':
    main()

# EOF
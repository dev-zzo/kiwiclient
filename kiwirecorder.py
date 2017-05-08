import codecs
import logging
import os
import struct
import sys
import time
from optparse import OptionParser

import kiwiclient

class KiwiRecorder(kiwiclient.KiwiSDRClientBase):
    def __init__(self, options):
        super(KiwiRecorder, self).__init__()
        self._options = options
        self._start_ts = None

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
        self._write_samples(samples)
        sys.stdout.write('\rBlock: %08x, RSSI: %-04d' % (seq, rssi))

    def _get_output_filename(self):
        ts = time.strftime('%Y%m%dT%H%MZ', self._start_ts)
        sta = '' if self._options.station is None else '_' + self._options.station
        return '%s_%d_%s%s.wav' % (ts, int(self._options.frequency * 1000), self._options.modulation, sta)

    def _write_samples(self, samples):
        """Output to a file on the disk."""
        now = time.gmtime()
        if self._start_ts is None or self._start_ts.tm_hour != now.tm_hour:
            self._start_ts = now
            # Write a static WAV header
            with open(self._get_output_filename(), 'wb') as fp:
                fp.write(struct.pack('<4sI4s', 'RIFF', 0x7FFFFFFF, 'WAVE'))
                fp.write(struct.pack('<4sIHHIIHH', 'fmt ', 16, 1, 1, int(self._sample_rate), int(self._sample_rate) * 16 / 8, 16 / 8, 16))
                fp.write(struct.pack('<4sI', 'data', 0x7FFFFFFF))
            print "Started a new file."
        with open(self._get_output_filename(), 'ab') as fp:
            # TODO: something better than that
            samples.tofile(fp)


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
        recorder = KiwiRecorder(options)
        
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
            print repr(e)
            break


if __name__ == '__main__':
    main()

# EOF
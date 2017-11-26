# UK
HOST_DRM = southwest.ddns.net
HOST_DRM_PORT = 8073
FREQ_DRM = 3965

drm:
	python kiwirecorder.py -s $(HOST_DRM) -p $(HOST_DRM_PORT) -f $(FREQ_DRM) -m iq -L -5000 -H 5000

# UK
HOST_FAX = southwest.ddns.net
HOST_FAX_PORT = 8073
FREQ_FAX = 2618.5

# Australia
#HOST_FAX = sdrtas.ddns.net
#HOST_FAX_PORT = 8073
#FREQ_FAX = 13920

fax:
	python kiwifax.py -s $(HOST_FAX) -p $(HOST_FAX_PORT) -f $(FREQ_FAX) -F
#	python kiwifax.py -s $(HOST_FAX) -p $(HOST_FAX_PORT) -f $(FREQ_FAX) -F --iq-stream

help:
	python kiwifax.py --help
	@echo
	python kiwirecorder.py --help

clean:
	-rm -f *.log *.wav *.png

clean_dist: clean
	-rm -f *.pyc */*.pyc

# DRM
# UK
HOST_DRM = southwest.ddns.net
HOST_DRM_PORT = 8073
FREQ_DRM = 3965

drm:
	python kiwirecorder.py -s $(HOST_DRM) -p $(HOST_DRM_PORT) -f $(FREQ_DRM) -m iq -L -5000 -H 5000


# FAX
# UK
#HOST_FAX = southwest.ddns.net
#HOST_FAX_PORT = 8073
#FREQ_FAX = 2618.5

# Australia
HOST_FAX = sdrtas.ddns.net
HOST_FAX_PORT = 8073
FREQ_FAX = 13920

fax:
	python kiwifax.py -s $(HOST_FAX) -p $(HOST_FAX_PORT) -f $(FREQ_FAX) -F
#	python kiwifax.py -s $(HOST_FAX) -p $(HOST_FAX_PORT) -f $(FREQ_FAX) -F --iq-stream


# Two IQ servers recording to two files in parallel
HOST_IQ1 = fenu-radio.ddns.net
HOST_IQ2 = hb9ryz.homeip.net

two:
#	python kiwirecorder.py -2 --s1=$(HOST_IQ1) --f1=77.5 --s2 $(HOST_IQ2) --f2=60 -m iq -L -5000 -H 5000
#   with "station ID" info in file name
	python kiwirecorder.py -2 --s1=$(HOST_IQ1) --f1=77.5 --station=DCF77 --s2 $(HOST_IQ2) --f2=60 --station2=MSF -m iq -L -5000 -H 5000
#   single frequency used by both servers
#	python kiwirecorder.py -2 --s1=$(HOST_IQ1) --f1=77.5 --station=DCF77 --s2=$(HOST_IQ2) -m iq -L -5000 -H 5000


help:
	python kiwifax.py --help
	@echo
	python kiwirecorder.py --help

clean:
	-rm -f *.log *.wav *.png

clean_dist: clean
	-rm -f *.pyc */*.pyc

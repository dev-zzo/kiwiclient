## -*- octave -*-

function [x,xx]=proc(fn) ##20171119T171520Z_30000_iq.wav
  tic
  [x,fs]=read_kiwi_iq_wav(fn);
  toc;

  idx = find(cat(1,x.gpslast)<6);
  xx  = {};
  for i=1:length(idx)
    j=idx(i);
    xx(i).t = x(j).gpssec + 1e-9*x(j).gpsnsec + [0:length(x(j).z)-1]'/fs;
    xx(i).z = x(j).z;
  end

  if length(xx) != 0
    subplot(2,2,1:2); plot(mod(cat(1,xx.t), 1), abs(cat(1,xx.z)));                                  xlabel("GPS seconds mod 1 (sec)");
    subplot(2,2,3);   plot(mod(cat(1,xx.t), 1), abs(cat(1,xx.z)), '.'); xlim([0.0285 0.0294]) ;     xlabel("GPS seconds mod 1 (sec)");
    subplot(2,2,4);   plot(mod(cat(1,xx.t), 1), abs(cat(1,xx.z)), '.'); xlim(0.1+[0.0285 0.0294]) ; xlabel("GPS seconds mod 1 (sec)");
  end
endfunction

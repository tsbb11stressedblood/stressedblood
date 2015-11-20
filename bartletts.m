function [rx,n] = bartletts(x)
  %Bartlett for ACF
L = numel(x);


rx_2 = zeros(1, L);
rx = zeros(1, 2*L-1);

for k=[1:L],
  for n=[1:L-abs(k)-1],
      rx_2(k) = rx_2(k) + x(n+abs(k)-1).*x(n);
  end
end

rx_2 = (1/L)*rx_2;

rx_1 = fliplr(rx_2);

rx = [rx_1(1:end-1) rx_2];
n=1:L;

end
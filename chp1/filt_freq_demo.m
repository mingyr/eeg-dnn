x = 1 : 256;
y = sin(0.1 * x) + 0.1 * sin(10 * x);
subplot(2, 2, 1);
plot(x, y);

Y = fft(y);
Y = fftshift(Y);
Y_mag = abs(Y);
subplot(2, 2, 2);
plot(x, Y_mag);

[v, i] = max(Y_mag);
[v, j] = max(flip(Y_mag));
j = length(Y_mag) - j + 1;

Y(1 : (i-10)) = 0i;
Y((j+10) : end) = 0i;

f = ones(size(x));
f(1 : (i-10)) = 0;
f((j + 10) : end) = 0;
subplot(2, 2, 3);
plot(x, f);

Y = ifftshift(Y);
y = ifft(Y);
subplot(2, 2, 4);
plot(x, y);

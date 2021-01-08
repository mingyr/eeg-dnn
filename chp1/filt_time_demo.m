x = (1:256);
subplot(2, 2, 1);
y = sin(0.1 * x) + 0.1*sin(10*x);
plot(x, y);
title('');

f = 1/3* [1, 1, 1];
x_tmp = [1, 2, 2, 3, 4, 4, 5];
y_tmp = [0, 0, 1/3, 1/3, 1/3, 0, 0];
subplot(2, 2, 2);
line(x_tmp, y_tmp);

F = freqz(f);
subplot(2, 2, 3);
plot(abs(F));
title('');

Y = conv(y, f, 'same');
subplot(2, 2, 4);
plot(x, Y);
title('');


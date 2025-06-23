%% Kompleksna stevila in kompleksna eksponentna funkcija
Fs = 100; % Vzorcevlanka frekfenca [Hz]
T = 1; % Dolzina intervala [s]
t = linspace(0, T, Fs); % Veckotr casovnih tocks

% Paramteri kompleksne eksponentne funckije e^(j * (2pi * f1 * t + p));
f1 = 5; % frekfenca [Hz]
p1 = 0; % faza [rad]
A1 = 0.5; % Amplituda

s1 = A1 * exp(1j * (2*pi*t*f1 + p));

figure;
% Prikaz relane komponente
subplot(3, 2, 1);
plot(t, real(s1), 'LineWidth', 1, 'Color', 'r');
xlabel("Time [s]");
ylabel("Real values");
title("Realna komponenta kompleksne eksponenten funkcije");

subplot(3, 2, 2);
hold on
for n = 1:10
    plot(t(n), real(s1(n)), 'ro', 'MarkerSize', 5);
    text(t(n), real(s1(n)) + 0.05, [num2str(n)]);
end
xlabel("Time [s]");
ylabel("Real values");
title("Prvih 10 vrednosti realna komponente");
hold off

% Prikaz imaginarne komponente
subplot(3, 2, 3);
plot(t, imag(s1), 'LineWidth', 1);
title("Imaginarna komponenta kompleksne eksponente funckije");
xlabel("Time [s]")
ylabel("Imag. values");

subplot(3, 2, 4);
hold on
for n = 1:10
    plot(t(n), imag(s1(n)), "bo", 'MarkerSize', 5);
    text(t(n), imag(s1(n)) + 0.05, [num2str(n)]);
end
title("Prvih 10 vrednosti imaginarne komponente");
xlabel("Time [s]");
ylabel("Imag. values");
hold off

% Priakz skupnega vektorskega prostora imaginarne in realne komponente
subplot(3, 2, 5);
plot(real(s1), imag(s1), 'LineWidth', 1, 'Color', 'g');
title("Skupen vektorski prostor imaginarne in realne komponente - Kompleksna ravnina");
xlabel("Realna os");
ylabel("Imaginarna os");

subplot(3, 2, 6);
hold on
for n = 1: 10
    plot(real(s1(n)), imag(s1(n)), 'o');
    text(real(s1(n)), imag(s1(n)) + 0.05, [num2str(n)]);
end
xlabel("Realna os");
ylabel("Imaginarna os");
hold off

% Vprasanja
% 1) V kaksem razmerju sta imaginarna in realna komponenta?
% 2) Kako sta imaginarna in realna komponenta izrazeni?
% 3) Kaj pocnemo z spreminjanjem faze?
% 4) Kaksem vpliv ima spreminjanje predznaka (-/+) i/j?
% 5) Kaksen vplig ima faza ce je oblika formule : s1 = A1 * exp(1j * (2*pi*t*f1) + phase)

% Odgovori
% 1, 2) realna komponenta je cos imaginaran komponeta pa sin (sta ortogonalni)
% Iz primerov je razvidno, da se skupen vekotrski prostor krog

% 3) z spreminjanjem faze spreminjamo zacetno tocko rotiranja

% 4) prednzak (-/+) j vpliva na smer rotacije

% 5)v s1 = A1 * exp(1j * (2*pi*t*f1) + phase) (ki je napacna), faza deluje kot amplituda


%% Kompleksne sinusoide z konstantno amplitudo v 3D prostori
Fs = 100;
T = 1;
t = linspace(0, T, Fs);

% Parametri kompleksne sinusoide
f = 5;
p = 0;
A = 1;

s = exp(1j * (2 * pi * t * f + p));

figure;
hold on;
plot3(t, real(s), imag(s), 'bo-', 'LineWidth', 1);
plot3(t(1), real(s(1)), imag(s(1)), 'ro', 'MarkerSize',10, 'MarkerFaceColor', 'r'); % izpostavi zacetno tocko
hold off
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(['Kompleksna eksponenta funkcija : frek = ' num2str(f) ' Hz, faza = ' num2str(p) ' rad']);

%% Kompleksne sinusoide z nestacionarno aplitudo
Fs = 100;
T = 1;
t = linspace(0, T, Fs);

% Parametri kompleksne sinusoide
f = 5;
p = pi / 2;
A = t + 1; % Amplituda se veca skozi cas

s = A .* exp(1j * (2 * pi * t * f + p)); % .* je mnozenje po elementih (potrebno ker sta A in t oba vektorja in bi klasicni * vrnil vektorski produkt)

figure;
hold on;
plot3(t, real(s), imag(s), 'bo-', 'LineWidth', 1);
plot3(t(1), real(s(1)), imag(s(1)), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold off;
title(['Kompleksna sinusoida z spreminjajoco Amplitudo. frek = ' num2str(f) ' Hz in faza = ' num2str(p) ' rad']);
xlabel('Time [s]');
ylabel('Real axis');
zlabel('Imag. axis');
%% Dve kompleksni funkciji in njuna ortogonalnost
Fs = 1000;
T = 1;
t = linspace(0, T, Fs);
N = T * Fs; % stevilo tock na opazovalnem intervalu

% Parametri 1. kompleksne sinusoide
f1 = 5;
p1 = 0;
A1 = 1;

s1 = A1 * exp(1j * (2 * pi * t * f1 + p1));

% Parametri 2. kompleksne sinusoide
f2 = 6;
p2 = 0;
A2 = 1;

s2 = A2 * exp(1j * (2 * pi * t * f2 + p2));

figure;
subplot(2, 2, 1);
hold on; grid on;
plot3(t(1), real(s1(1)), imag(s1(1)), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
plot3(t, real(s1), imag(s1), "ro-", 'MarkerSize', 1, 'LineWidth', 1);
title(['Kompleksna sinusoida s frek ' num2str(f1) ' Hz, faza ' num2str(p1) ' rad in amplituda ' num2str(A1)]);
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
axis([0,T,-max(A1,A2),max(A1,A2),-max(A1,A2),max(A1,A2)]);
hold off;

subplot(2, 2, 2);
hold on; grid on;
plot3(t(1), real(s2(1)), imag(s2(1)), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
plot3(t, real(s2), imag(s2), "bo-", 'MarkerSize', 1, 'LineWidth', 1);
axis([-Fs * T * A1 * A2, Fs * T * A1 * A2, -Fs * T * A1 * A2, Fs * T * A1 * A2]);
title(['Kompleksna sinusoida s frek ' num2str(f2) ' Hz, faza ' num2str(p2) ' rad in amplituda ' num2str(A2)]);
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
axis([0,T,-max(A1,A2),max(A1,A2),-max(A1,A2),max(A1,A2)]);
hold off;

% Prikaz zmnozkov realne in imaginarne osi - v skupnem vektorskem prostoru
% tovori krog - zmonzek vrednosti realne osi je funkcija cos, zmnozek vrednosti imaginarne osi je funckija sin
% torej bi sestevek (skalarni produkt) bil enak 0 (Funkciji sta ortogonalni)
subplot(2, 2, 3);
hold on; grid on;
plot3(t, real(s1 .* conj(s2)), imag(s1 .* conj(s2)), 'LineWidth', 1, 'Color', 'magenta');
title(["Zmonzek elementov imaginarne in realne osi funkcij" 
       newline  'f1: ' num2str(f1) ' Hz, faza ' num2str(p1) ' rad in amplituda ' num2str(A1) 
       newline 'f2: ' num2str(f2) ' Hz, faza ' num2str(p2) ' rad in amplituda ' num2str(A2)]);
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
axis([0,T,-abs(A1)*abs(A2),abs(A1)*abs(A2),-abs(A1)*abs(A2),abs(A1)*abs(A2)]);
hold off;

% Prikaz skalarnega producta

subplot(2, 2, 4);
hold on; grid on;
sp = sum(s1 .* conj(s2));
plot([0, real(sp)], [0, imag(sp)], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
plot([real(sp)], [imag(sp)], 'ro-', 'MarkerFaceColor', 'r');
title(['Skalarni produkt f1 in f2 v kompleksni ravnini']);
xlabel('Real axis');
ylabel('Imag. axis');
axis([-Fs * T * A1 * A2, Fs * T * A1 * A2, -Fs * T * A1 * A2, Fs * T * A1 * A2]);
hold off;


% Vrpasanja
% 1) Kaj se dogaja, ko sta f1 in f2 razlicni za en nihaj na opazovalnem intervalu, kako nato vpliva faza?
% 2) Kaj se dogaja, ko sta f1 in f2 razlicni za veckratnik nihaja na opazovalnem intervalu, kako nato vpliva faza?
% 3) Kaj se dogaja, ko sta f2 in f2 razlicna za poljuben del nihaja na opazovalnem intervalu in kako na to vpliva faza?

% Odgovori
% 1) Ce se f1 in f2 razlikujeta za tocno en nihaj na opazovalnem intervalue bi v skupnem vek.
% prostoru zmonskov imaginarnih komponent in realnih komponet videli 1
% krog. Razlika faz signalov, bi nam povedala, zacetno tocko tega krozenja - torej nima vpliva na ortogonalnost.

% 2) Ce se f1 in f2 razlikujeta za celostevilski veckratnih "n" nihaja na opazovalenm intervalu bi v skupnem vek.
% prostoru zmonskov imaginarnih komponent in realnih komponet videli "n"
% krogov. Razlika faz signalov, bi nam povedala, zacetno tocko tega krozenja - torej nima vpliva na ortogonalnost.

% 3) Ce se f1 in f2 ne razlikujeta za n * 1/T kjer je "n" celo stevilo, bo to povzorcilo, da v kompleksni ravnini zadnji krog nebo cel/koncan
% (ce bi bila razlika npr. 0.5 bi dobili pol krog), kar pomeni da sinusoidi nebosta ortogonalni.
% Razlika faz signalov, bi nam povedala, zacetno tocko tega krozenja - torej nima vpliva na ortogonalnost.



%% Vpliv faznega premika na kompleksni sinusoidi z enakima frekfencama
% Vprasanja
% 1) Kaj se dogaja, ko sta f1 in f2 enaki, fazi sinusoid pa sta razlicni?

% Odgovor
% 1) Ce si sklarni produkt sp = <s1, s2> zamislimo kot vektor iz izhodica,
% ga spreminjanje faze rotira za <faza1 - faza2> radianov. Torej
% povzroci samo spremebno pozicije skalarnega produkta v kompleksni ravnini

Fs = 1000;
T = 1;
t = linspace(0, T, Fs);

A1 = 1; A2 = 2;
f1 = 5; f2 = 5;

p1 = 0;
s1 = A1 * exp(1j * (2 * pi * t * f1 + p1));

figure;
hold on; grid on;
for p2 = linspace(0, pi/2, 20)
    s2 = A2 * exp(1j * (2 * pi * t * f2 + p2));
    sp = sum(s1 .* conj(s2));

    plot([0, real(sp)], [0, imag(sp)], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
    plot([real(sp)], [imag(sp)], 'ro-', 'MarkerFaceColor', 'r');
    text(real(sp), imag(sp), ['diff = ' num2str(round((p1-p2)/pi, 2)) 'pi']);

end
title('Vpliv razlike faz (diff) sinusoid na skalarni produkt');
xlabel('Real axis');
ylabel('Imag. axis');
axis([-Fs * T * A1 * A2, Fs * T * A1 * A2, -Fs * T * A1 * A2, Fs * T * A1 * A2]);
hold off;

%% Orotogonalnost realne in kompleksne sinusoida
Fv = 1000;
T = 1;
t = linspace(0, T, Fs);
N = T * Fv; % stevilo vzorcev na opazovalnem intervalu

% parametri relane sinusoide
f1 = 6;
p1 = 0;
A1 = 1;
s1 = A1 * sin(2 * pi * t * f1 + p1);

% kompleksna sinusoida
f2 = 5;
p2 = 0;
A2 = 1;
s2 = A2 * exp(j * (2 * pi * t * f2 + p2));

figure;

subplot(2, 2, 1);
hold on;
plot3(t, real(s1), imag(s1), "r-", "MarkerSize", 1);
plot3(t(1), real(s1(1)), imag(s1(1)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(['Realna funkcija ' num2str(f1) 'Hz, faza ' num2str(round((p1 / pi),2)) 'pi rad']);
axis([0,T,-max(A1,A2),max(A1,A2),-max(A1,A2),max(A1,A2)]);
hold off;

subplot(2, 2, 2);
hold on;
plot3(t, real(s2), imag(s2), "b-", "MarkerSize", 1);
plot3(t(1), real(s2(1)), imag(s2(1)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(['Eksponenta kompleknsa funkcija ' num2str(f2) 'Hz, faza ' num2str(round((p2 / pi),2)) 'pi rad']);
axis([0,T,-max(A1,A2),max(A1,A2),-max(A1,A2),max(A1,A2)]);
hold off;

subplot(2, 2, 3);
plot3(t, real(s1 .* conj(s2)), imag(s1 .* conj(s2)), '-', 'LineWidth', 1, 'MarkerFaceColor', 'magenta');
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(["Zmonzek elementov imaginarne in realne osi funkcij" 
       newline  'f1: ' num2str(f1) ' Hz, faza ' num2str(round(p1 / pi, 2)) 'pi rad' 
       newline 'f2: ' num2str(f2) ' Hz, faza ' num2str(round(p2 / pi, 2)) 'pi rad']);
axis([0,T,-abs(A1)*abs(A2),abs(A1)*abs(A2),-abs(A1)*abs(A2),abs(A1)*abs(A2)]);
hold off;

subplot(2, 2, 4);
hold on; grid on;
sp = sum(s1 .* conj(s2));
plot([0, real(sp)], [0, imag(sp)], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
plot([real(sp)], [imag(sp)], 'ro-', 'MarkerFaceColor', 'r');
title('Skalarni produkt f1 in f2 v kompleksni ravnini');
xlabel('Real axis');
ylabel('Imag. axis');
axis([-Fs * T * A1 * A2, Fs * T * A1 * A2, -Fs * T * A1 * A2, Fs * T * A1 * A2]);
hold off;

% Vrpasanja
% 1) Kaj se dogaja, ko sta f1 in f2 enaki, fazi sinusoid pa sta razlicni?
% 2) Kaj se dogaja, ko sta f1 in f2 razlicni za en nihaj na opazovalnem intervalu, kako nato vpliva faza?
% 3) Kaj se dogaja, ko sta f1 in f2 razlicni za veckratnik nihaja na opazovalnem intervalu, kako nato vpliva faza?
% 4) Kaj se dogaja, ko sta f2 in f2 razlicna za poljuben del nihaja na opazovalnem intervalu in kako na to vpliva faza?

% Odgovori
% 1) Ko velja f1 = f2 in faz1 != faz2 ima to samo vpliv da spremeni pozicijo skalarnega produkta v kompleksni ravnini.
% Jo zarotira za (faz1 - faz2) rad
% Skalarni produkt je maksimalen

% 2) Ce velja (f1 - f2) = 1/T sta signala ortogonalno. 
% Faza nima vpliva na ortogonalnost ampak samo doloca zacetno pozicijio krozenja. 
% V kompleksni ravnini zmnozkov vidim kroznice, ki skupaj tvorujio obliko  enega kroga
% vidimo 1 popolen obhod po kroznici

% 3) Ce velja (f1 - f2) = (1/T) * n, kjer je "n" celo stevilo sta signala ortogonalno. 
% Faza nima vpliva na ortogonalnost ampak samo doloca zacetno pozicijio krozenja. 
% V kompleksni ravnini zmnozkov vidim kroznice, ki skupaj tvorujio obliko n-celih krogov 
% vidimo "n" popolnih obhodov po korznici

% 4) Ce velja (f1 - f2) != (1/T) * n, kjer je "n" celo stevilo, potem signala nista ortogonalna. 
% Faza nima vpliva na ortogonalnost ampak samo doloca zacetno pozicijio krozenja. 
% V kompleksni ravnini zmnozkov vidim kroznice, ki skupaj tvorijo obliko ne celega stevila krogov
% vidimo ne popolen obhod po kroznici

%% Orotogonalnost dveh realnih sinusoid
Fv = 1000;
T = 1;
t = linspace(0, T, Fs);
N = T * Fv; % stevilo vzorcev na opazovalnem intervalu

% parametri 1. relane sinusoide
f1 = 5;
p1 = 0;
A1 = 1;
s1 = A1 * sin(2 * pi * t * f1 + p1);

% parametri 2. realne sinusoida
f2 = 5;
p2 = pi/4;
A2 = 1;
s2 = A2 * sin(2 * pi * t * f2 + p2);

figure;
subplot(2, 2, 1);
hold on;
plot3(t, real(s1), imag(s1), "r-", "MarkerSize", 1);
plot3(t(1), real(s1(1)), imag(s1(1)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(['Realna funkcija ' num2str(f1) 'Hz, faza ' num2str(round((p1 / pi),2)) 'pi rad']);
axis([0,T,-max(A1,A2),max(A1,A2),-max(A1,A2),max(A1,A2)]);
hold off;

subplot(2, 2, 2);
hold on;
plot3(t, real(s2), imag(s2), "b-", "MarkerSize", 1);
plot3(t(1), real(s2(1)), imag(s2(1)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(['Realna kompleknsa funkcija ' num2str(f2) 'Hz, faza ' num2str(round((p2 / pi),2)) 'pi rad']);
axis([0,T,-max(A1,A2),max(A1,A2),-max(A1,A2),max(A1,A2)]);
hold off;

subplot(2, 2, 3);
plot3(t, real(s1 .* conj(s2)), imag(s1 .* conj(s2)), '-', 'LineWidth', 1, 'MarkerFaceColor', 'magenta');
xlabel("Time [s]");
ylabel("Real axis");
zlabel("Imag. axis");
title(["Zmonzek elementov imaginarne in realne osi funkcij" 
       newline  'f1: ' num2str(f1) ' Hz, faza ' num2str(round(p1 / pi, 2)) 'pi rad' 
       newline 'f2: ' num2str(f2) ' Hz, faza ' num2str(round(p2 / pi, 2)) 'pi rad']);
axis([0,T,-abs(A1)*abs(A2),abs(A1)*abs(A2),-abs(A1)*abs(A2),abs(A1)*abs(A2)]);
hold off;

subplot(2, 2, 4);
hold on; grid on;
sp = sum(s1 .* conj(s2));
plot([0, real(sp)], [0, imag(sp)], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
plot([real(sp)], [imag(sp)], 'ro-', 'MarkerFaceColor', 'r');
title('Skalarni produkt f1 in f2 v kompleksni ravnini');
xlabel('Real axis');
ylabel('Imag. axis');
axis([-Fs * T * A1 * A2, Fs * T * A1 * A2, -Fs * T * A1 * A2, Fs * T * A1 * A2]);
hold off;

% Vrpasanja
% 1) Kaj se dogaja, ko sta f1 in f2 enaki, fazi sinusoid pa sta razlicni?
% 2) Kaj se dogaja, ko sta f1 in f2 razlicni za en nihaj na opazovalnem intervalu, kako nato vpliva faza?
% 3) Kaj se dogaja, ko sta f1 in f2 razlicni za veckratnik nihaja na opazovalnem intervalu, kako nato vpliva faza?
% 4) Kaj se dogaja, ko sta f2 in f2 razlicna za poljuben del nihaja na opazovalnem intervalu in kako na to vpliva faza?

% Odgovori
% 1) Kadar je f1 = f2 in faz1 != faz2 ne moremo biti prepricani, da bo
% skalarni produkt maksimalen oz. vecji od 0. Z faznim premikom 
% sinusoid lahko namrec dosezemo ortogonalnost med njima (ce bo faz1 - faz2 = n * (pi/2), n -> celo stevilo ).
% Npr. ce imamo 2 sin signala z 5Hz ce enemu dolocimo fazo pi/2 (postance cos) bi skalarni produkt bil
% enak 0 kljub temu, da gre za 2 signala z enako frekfenco.

% 2) Kadar je (f1 - f2) = 1/T in faz1 = faz2 bota signala vedno
% ortogonalna. Faza nebo vplivala na ortogonalnost amapk samo dolocala
% zacetno pozicijo krozenja. Imeli bomo 1 poln obhod po kroznici (1 polna perioida)

% 3) Kadar je (f1 - f2) = n * (1/T), n -> celo stevilo in faz1 = faz2 bota signala vedno
% ortogonalna. Faza nebo vplivala na ortogonalnost amapk samo dolocala
% zacetno pozicijo krozenja. Imeli bomo n polnih obhod po kroznici (n polnih perioid)

% 4) Kadar (f1 - f2) != n * (1/T), n -> celo stevilo in faz1 = faz2 signala nikoli ne bosta
% ortogonalna. Faza nebo vplivala na ortogonalnost amapk samo dolocala
% zacetno pozicijo krozenja. Imeli bomo ne celo stevilo obhodov po kroznici (nimamo cele periode)

% DFT torej potrebuje 1 kompleksno eksponento funkcijo, da lahko enolicno doloci ali
% je frekfenca prisotna v originalnem signalu brez, da bi skrebel na vpliv
% faze. Kompleksna eksponenta funkcija je namrec sestavljena iz sin iz cos. Torej
% bomo lahko zaznavali, tako spremebe v fazi kot amplitudi

%% Fourierova analiza po domace (samo polovicno ali pa se manj)
Fs = 1000;
T = 1;
t = 0:(1/Fs):(T - 1/Fs); % starn : step:  end

% 1. REALNA sinusoida
A1 = 1;
f1 = 5;
p1 = 0;
s1 = A1 * sin(2 * pi * t * f1 + p1);

% 2. REALNA sinusoida
A2 = 1;
f2 = 6;
p2 = 0;
s2 = A2 * sin(2 * pi * t * f2 + p2);

% skupen signal
s3 = s1 + s2;

figure; hold on;
plot(t, s1, "bo-");
plot(t, s2, "ro-");
plot(t, s3, "ko-");

legend({'Sinusoida 1 (s1)','Sinusoida 2 (s2)','Vsota sinusoid (s3)'});
xlabel('Time [s]');
ylabel('Amplitude')
title(['Sinusoida 1: ' num2str(f1) 'Hz faza: ' num2str(round(p1/pi, 2)) 'pi rad' ...
    newline 'Sinusoida 1: ' num2str(f2) 'Hz faza: ' num2str(round(p2/pi, 2)) 'pi rad'])

hold off;

% skalarni produkt s*s1 -> s kaksno amplituod nastoma singal s1 v signalu s
sp1 = dot(s3, s1); % amplituda s katero je signal s1 prisoten v s3
sp2 = dot(s3, s2); % amplutida s katero je signal s2 prisoten v s3


s3_fft = fft(s3);
freqs = linspace(0, Fs, length(s3));
phases = atan2(imag(s3_fft), real(s3_fft)); % atan(y/x) = faza --> atan(sin(phi)/cos(phi));

figure;
% Amplitudni spekter frekfenc
subplot(2, 1, 1);
plot(freqs, abs(s3_fft));
title("Freq. Amplitude specter");
xlabel("Frequency [Hz]");
ylabel("Amplitude");

% Fazni spekter frekfenc
subplot(2, 1, 2);
plot(freqs, phases/pi, "Color", "magenta");
title("Freq. Phase specter");
xlabel("Frequency [Hz]");
ylabel("Phase [pi rad]");

% V faznem spektru prikazemo zacetno tocko/fazo posamezne frekfence (napram na cos)
% V nasem spektru vidimo, da sta sinusoidi 5hz in 6hz prisotni z fazo -pi/2,
% kljub temu, da sta sestavljeni kot A1*sin(2pi*t*f + 0). To je zato ker
% v faznem spektru fazo vedno gledamo na pram cos nihanju. Razlika med sin
% in cos pa je pi/2;
%
% Torej fazni spekter FFT se vedno nanasa na cos (realni del)

%% FOURIEROVA analiza v matlabu (fft)
Fs = 1000;
T = 0.02;
t = linspace(0, T, Fs);

% 1. sinusoida
f1 = 55;
p1 = 0;
A1 = 1;
s1 = A1*sin(2*pi*f1*t + p1);

% FFT
s1_fft = fft(s1);
freqs = linspace(0, Fs, length(s1_fft));
phases = atan2(imag(s1_fft), real(s1_fft));

figure;
subplot(4, 1, 1);
plot(t, s1);
title(['Orginalen signal ' num2str(f1) ' Hz faza ' num2str(p1)]);
xlabel("Time [s]");
ylabel("Amplitude");

% Amplitudni spekter frekfenc
subplot(4, 1, 2);
plot(freqs, abs(s1_fft));
title("Freq. Amplitude specter");
xlabel("Frequency [Hz]");
ylabel("Amplitude");

% Fazni spekter frekfenc
subplot(4, 1, 3);
plot(freqs, phases/pi, "Color", "magenta");
title("Freq. Phase specter");
xlabel("Frequency [Hz]");
ylabel("Phase [pi rad]");
ylim([-1.1 1.1]);


% Inverzni fft
subplot(4, 1, 4);
s1_reconstructed = ifft(s1_fft);
plot(t, s1_reconstructed);
title("Rekonstruiran signal");
xlabel("Time [s]");
ylabel("Amplitude");

% Vidimo lahko da je ampliudni spekter simetricen preko N/2 in da je fazni
% spekter antisimetricen preko N/2


%% Simulacija spektralnog razlivanja

Fs = 100;
T = 1;
t = 0:1/Fs:T;

A = 1;
p = 0;
f = (1/T) * 10.5; % (1/T) * n; ce "n" ni celo stevilo dobimo spektraln razlivanje (ni maksimalen skalarni produkt)

s = A * sin(2*pi*t*f + p);
freqs = linspace(0, Fs, length(s));
y = fft(s);

figure;
subplot(2, 1, 1);
hold on; grid on;
plot(t, s);
title(['Sin z ' num2str(f) ' Hz in fazo ' num2str(round(p/pi, 2)) 'pi rad']);
xlabel('Time [s]');
ylabel('Amplitude');
hold off;

subplot(2, 1, 2);
hold on; grid on;
stem(freqs(1:length(s)/2), abs(y(1:length(s)/2)), 'r');
xlabel('Frequency [Hz]');
ylabel('Amplitude');
hold off;


% To spektralenga razlivanja pride kadara |T| * f ni celo stevliska. 
% vrednost. To pomeni da je frekfenca taksna, da je dejansko ne moremo
% zaznati (sedi nekje med diskretnim korakom). To bo povzrocilo, da se bo
% nejn vpliv / energija razliza na sosednje frekfence (leve in desne)
% 1/T je frekfenci korak (delta frek.)

%% Bitna locljivost

Fs = 1000;
T = 1;
t = linspace(0, T, Fs);
f = 3;
A = 16;

b = 3; % bitna locljivost
step = (2*A)/(2^b);

s = round(A*sin(2*pi*f*t)/step) * step;

quantization_error = (A - (-A)) / 2^b; % (Vmax - Vmin)/2^n; Vmax = 16 (A), Vmin = -16(-A)

figure;
plot(t, s);
xlabel("Time [s]");
ylabel("Amplitude");
title(['Amplitude ' num2str(A) ' bit resolutuin ' num2str(b) ' napaka ' num2str(round(quantization_error, 2))]);

figure;
plot(t, s);
xlabel("Time [s]");
ylabel("Amplitude");
title(['Amplitude ' num2str(A) ' bit resolutuin ' num2str(b) ' napaka ' num2str(round(quantization_error, 2))]);

% Pogosto se uporablja bitna locljivost 16 bitov ali 24 bitov
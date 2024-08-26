clc
clear

tic
mT=.4;
beta = 1.5;
sT=.2*mT;
T=70;

N = 1000;

varT = sT.^2;
b=mT./varT;
a=mT.^2./varT/T^beta;


D = zeros(N,T);
to = zeros(1,N);
for i=1:N
    [mT*100 i/N*100]
    
%     to(i) = randi(T-1);
%     D(i,to(i)) = rand;
    to(i)=1;
    D(i,1:to(i)) = 0;

    for t = to(i):T

        dD = gamrnd(a*t^beta-a*(t-1)^beta,1/b);
        D(i,t+1) = D(i,t)+dD;
        D(i,t+1) = min(.9999999,D(i,t+1));

    end
end

hold off
plot(D')
hold all
plot(mean(D),'linewidth',4)

x=mean(D);
x(T)
x=std(D);
x(T)

bin_size = 2;
n_bins = 100/bin_size;
Dlabel=D;
for k=1:N

    for l=to(k):T+1
        for j=1:n_bins
            if D(k,l)<j*bin_size/100 && D(k,l)>=(j-1)*bin_size/100
                Dlabel(k,l) = j;
            end
        end
    end
end

trans_counts = zeros(n_bins,n_bins,T);

for i = 1:N
    for t = to(i)+1:T+1
        trans_counts(Dlabel(i,t-1),Dlabel(i,t),t-1)=1+trans_counts(Dlabel(i,t-1),Dlabel(i,t),t-1);
    end
end

counts = sum(trans_counts,2);

p = trans_counts;
for i = 1:n_bins
    for t = 2:T+1
        p(i,:,t-1)=trans_counts(i,:,t-1)/counts(i,1,t-1);
    end
end

toc

for i=1:70
    filename = ['p_trans', [int2str(i) '_' int2str(mT*100)], '.mat'];
    ppp=p(:,:,i);
    save(filename,'ppp');
end


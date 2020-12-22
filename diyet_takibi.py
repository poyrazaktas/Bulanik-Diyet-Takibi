import numpy as np

import matplotlib.pyplot as plt
# range ve çizimler değiştirilebilir

# OpenSource Üçgen Üyelik Fonksiyonu scikit :  https://github.com/scikit-fuzzy/scikit-fuzzy/tree/master/skfuzzy/membership


def trimf(x, abc):
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y

# OpenSource yamuk üyelik fonksiyonu scikit :  https://github.com/scikit-fuzzy/scikit-fuzzy/tree/master/skfuzzy/membership


def trapmf(x, abcd):
    assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
    a, b, c, d = np.r_[abcd]
    assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                          a <= b <= c <= d.'
    y = np.ones(len(x))

    idx = np.nonzero(x <= b)[0]
    y[idx] = trimf(x[idx], np.r_[a, b, b])

    idx = np.nonzero(x >= c)[0]
    y[idx] = trimf(x[idx], np.r_[c, c, d])

    idx = np.nonzero(x < a)[0]
    y[idx] = np.zeros(len(idx))

    idx = np.nonzero(x > d)[0]
    y[idx] = np.zeros(len(idx))

    return y

# OpenSource Üyelik Derecesi fonksiyonu scikit :  https://github.com/scikit-fuzzy/scikit-fuzzy/tree/master/skfuzzy


def interp_membership(x, xmf, xx):
    # Find the degree of membership ``u(xx)`` for a given value of ``x = xx``.
    # Nearest discrete x-values
    x1 = x[x <= xx][-1]
    x2 = x[x >= xx][0]

    idx1 = np.nonzero(x == x1)[0][0]
    idx2 = np.nonzero(x == x2)[0][0]

    xmf1 = xmf[idx1]
    xmf2 = xmf[idx2]

    if x1 == x2:
        xxmf = xmf[idx1]
    else:
        slope = (xmf2 - xmf1) / float(x2 - x1)
        xxmf = slope * (xx - x1) + xmf1

    return xxmf
# OpenSource Durulama scikit: https://github.com/scikit-fuzzy/scikit-fuzzy/tree/master/skfuzzy/defuzzify


def centroid(x, mfx):

    #    Defuzzification using centroid (`center of gravity`) method.

    sum_moment_area = 0.0
    sum_area = 0.0

    # If the membership function is a singleton fuzzy set:
    if len(x) == 1:
        return x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float)

    # else return the sum of moment*area/sum of area
    for i in range(1, len(x)):
        x1 = x[i - 1]
        x2 = x[i]
        y1 = mfx[i - 1]
        y2 = mfx[i]

        # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
        if not(y1 == y2 == 0.0 or x1 == x2):
            if y1 == y2:  # rectangle
                moment = 0.5 * (x1 + x2)
                area = (x2 - x1) * y1
            elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                moment = 2.0 / 3.0 * (x2-x1) + x1
                area = 0.5 * (x2 - x1) * y2
            elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                moment = 1.0 / 3.0 * (x2 - x1) + x1
                area = 0.5 * (x2 - x1) * y1
            else:
                moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                area = 0.5 * (x2 - x1) * (y1 + y2)

            sum_moment_area += moment * area
            sum_area += area

    return sum_moment_area / np.fmax(sum_area,
                                     np.finfo(float).eps).astype(float)


x_amr = np.arange(0, 5001, 1)
x_hedef = np.arange(-10000, 10001, 1)
x_kalori = np.arange(0, 101, 1)

amr_yavas = trapmf(x_amr, [-1, 0, 1000, 2500])
amr_orta = trimf(x_amr, [1500, 2500, 3500])
amr_hizli = trapmf(x_amr, [2500, 4000, 5000, 5001])

hedef_vermek = trapmf(x_hedef, [-10001, -10000, -8000, 3000])
hedef_almak = trapmf(x_hedef, [-3000, 8000, 10000, 10001])

kalori_az = trimf(x_kalori, [0, 0, 80])
kalori_fazla = trimf(x_kalori, [20, 100, 101])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6, 10))

ax0.plot(x_amr, amr_yavas, 'r', linewidth=2, label='AMH Yavaş')
ax0.plot(x_amr, amr_orta, 'g', linewidth=2, label='AMH Orta')
ax0.plot(x_amr, amr_hizli, 'b', linewidth=2, label='AMH Hızlı')
ax0.set_title('Metabolizma Hızı')
ax0.legend(loc='lower right')

ax1.plot(x_hedef, hedef_vermek, 'r', linewidth=2, label='Kilo Vermek')
ax1.plot(x_hedef, hedef_almak, 'g', linewidth=2, label='Kilo Almak')
ax1.set_title('Hedef Kilo')
ax1.legend(loc='lower right')

ax2.plot(x_kalori, kalori_az, 'r', linewidth=2, label='Az Kalorili')
ax2.plot(x_kalori, kalori_fazla, 'b', linewidth=2, label='Fazla Kalorili')
ax2.set_title('Kalori Tavsiyesi')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()

cinsiyet = int(input("Cinsiyetinizi giriniz (Erkek için 1, Kadın için 2): "))

if cinsiyet not in [1, 2]:
    print("Cinsiyet seçimi hatalı!")
    exit()

yas = int(input("Yaşınızı giriniz: "))
if yas < 0:
    print("Yaş değeri geçersiz!")
    exit()

kilo = float(input("Kilonuzu giriniz (kg): "))
if kilo < 0:
    print("Kilo değeri geçersiz!")
    exit()

boy = float(input("Boyunuzu giriniz (cm) : "))
if boy < 0:
    print("Boy değeri geçersiz!")
    exit()

aktivite = int(input("Haftalık aktiviteni değerlendir (1-5 arasında) : "))
if aktivite not in [1, 2, 3, 4, 5]:
    print("Aktivite değeri geçersiz!")
    exit()
hedef_kilo = float(input("Hedef kilonuzu giriniz (kg) : "))
if abs(kilo-hedef_kilo) > 10:
    print("Haftada 10 kilodan fazla vermek zararlı!")
    exit()


# bazal metabolizma hızı hesabı :
bmr = 0
if cinsiyet == 1:
    bmr = 66.5 + (13.75*kilo) + (5.0003*boy) - (6.755*yas)
elif cinsiyet == 2:
    bmr = 655 + (9.563*kilo) + (1.850*boy) - (4.676*yas)
else:
    print("hata var")


# aktivite çarpanı : amr hesabı için
# amr : active metabolic rate = bmr * aktivite

amr = 0
if aktivite == 1:
    amr = bmr*1.2
elif aktivite == 2:
    amr = bmr*1.375
elif aktivite == 3:
    amr = bmr*1.55
elif aktivite == 4:
    amr = bmr*1.725
elif aktivite == 5:
    amr = bmr*1.9
else:
    print("Girilen aktivite değerinde hata var")
    exit()
print("Aktif Metabolik Hız: ", amr)

# öğün tablodan da alınabilir
#ogun = int(input("Kaç kalori yediğinizi giriniz (kcal) : "))

# üyelik dereceleri hesabı için
input_amr = amr
input_hedef = (hedef_kilo-kilo)*1000

# üyelik dereceleri
amr_fit_yavas = interp_membership(x_amr, amr_yavas, input_amr)
amr_fit_orta = interp_membership(x_amr, amr_orta, input_amr)
amr_fit_hizli = interp_membership(x_amr, amr_hizli, input_amr)

hedef_fit_vermek = interp_membership(x_hedef, hedef_vermek, input_hedef)
hedef_fit_almak = interp_membership(x_hedef, hedef_almak, input_hedef)

# kurallar
kural1 = np.fmin(np.fmin(amr_fit_hizli, hedef_fit_almak), kalori_fazla)
kural2 = np.fmin(np.fmin(amr_fit_yavas, hedef_fit_vermek), kalori_az)
kural3 = np.fmin(np.fmin(amr_fit_orta, hedef_fit_almak), kalori_fazla)
kural4 = np.fmin(np.fmin(amr_fit_orta, hedef_fit_vermek), kalori_az)
kural5 = np.fmin(np.fmax(amr_fit_yavas, hedef_fit_almak), kalori_fazla)
kural6 = np.fmin(np.fmax(amr_fit_hizli, hedef_fit_vermek), kalori_az)
# çıkışlar
out_az = np.fmax(kural2, kural4, kural6)
out_fazla = np.fmax(kural1, kural3, kural5)

# veri görselleştirme: Değerlendirme çıkışı
kalori0 = np.zeros_like(x_kalori)

fig, ax3 = plt.subplots(figsize=(7, 4))
ax3.fill_between(x_kalori, kalori0, out_az, facecolor='r', alpha=0.7)
ax3.plot(x_kalori, kalori_az, 'r', linestyle='--')
ax3.fill_between(x_kalori, kalori0, out_fazla, facecolor='b', alpha=0.7)
ax3.plot(x_kalori, kalori_fazla, 'b', linestyle='--')
plt.tight_layout()
plt.show()

# durulaştırma
out_kalori = np.fmax(out_az, out_fazla)
defuzzied = centroid(x_kalori, out_kalori)  # değiştirilebilir
result = interp_membership(x_kalori, out_kalori, defuzzied)

# sonuç
coefficent = 0
durulama_sonuc = 0
r = result * 100
if r == 50:
    durulama_sonuc = 0
elif input_hedef>0:
    coefficent = abs((r-50))/100
    durulama_sonuc = amr+(coefficent*amr)
elif input_hedef<0:
    coefficent = abs((r-50))/100
    durulama_sonuc = amr - (coefficent*amr)



print("Result:", result)
print("Kalori Değerlendirme Çıkış Değeri (Crisp Value): ", r)

if durulama_sonuc == 0:
    print("Herhangi bir sapmaya gerek yok")
elif durulama_sonuc < amr:
    print("Beslenmenizi", durulama_sonuc, "kaloriye azaltmanız gerekir.")
else:
    print("Beslenmenizi", durulama_sonuc, "kaloriye arttırmanız gerekir.")
# Veri görselleştirme
fig, ax4 = plt.subplots(figsize=(7, 4))

ax4.plot(x_kalori, kalori_az, 'r', linewidth=0.5, linestyle='--')
ax4.plot(x_kalori, kalori_fazla, 'b', linewidth=0.5, linestyle='--')
ax4.fill_between(x_kalori, kalori0, out_kalori, facecolor='Orange', alpha=0.7)
ax4.plot([defuzzied, defuzzied], [0, result], 'k', linewidth=1.5, alpha=0.9)
ax4.set_title("Ağırlık Merkezi ile Durulaştırma")

plt.tight_layout()
plt.show()

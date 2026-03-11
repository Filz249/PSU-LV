# Zadatak 3
# Napišite program koji od korisnika zahtijeva unos brojeva u beskonačnoj petlji sve dok korisnik ne upiše „Done“ (bez navodnika). 
# Pri tome brojeve spremajte u listu. Nakon toga potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu vrijednost. 
# Sortirajte listu i ispišite je na ekran.
# Dodatno: osigurajte program od pogrešnog unosa (npr. slovo umjesto brojke) na način da program zanemari taj unos i ispiše odgovarajuću poruku.

from typing import Counter


list = []

while True:
    unos = input("Unos brojeva: ")
    if unos == str("Done" or "done"):
        break
    else:
        cnt = Counter(unos)
        list.append(unos)

print("List: ", list)
print("Unesenih brojeva:", cnt)
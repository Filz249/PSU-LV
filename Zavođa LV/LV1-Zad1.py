# Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je plaćen po radnom satu. 
# Koristite ugrađenu Python metodu input(). 
# Nakon toga izračunajte koliko je korisnik zaradio i ispišite na ekran. 
# Na kraju prepravite rješenje na način da ukupni iznos izračunavate u zasebnoj funkciji naziva total_euro.
# Primjer:
# Radni sati: 35 h
# eura/h: 8.5
# Ukupno: 297.5 eura

print("Unos radnih sati: ")
hours = float(input())
print("Unos plaće po satu: ")
pay = float(input())

ukupno = hours * pay

def total_pay(hours,pay):
    ukupno = hours * pay
    return ukupno

print("Rani sati: ", hours)
print("Eura/h: ", pay)
print("Ukupno: ", ukupno)
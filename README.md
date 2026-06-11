# Simple Song Maker (SSM) - AI Bahov Kompozitor

Projekat dubokog učenja (deep learning) koji koristi prilagođenu LSTM neuronsku mrežu za generisanje originalnih, četvoroglasnih polifonih kompozicija u stilu Johana Sebastijana Baha.

Model je treniran na `music21` korpusu Bahovih korala kako bi naučio složene harmonijske i melodijske strukture. Automatski orkestrira generisane sekvence za klasični gudački kvartet, kreirajući i MIDI fajlove i sintetizovani WAV audio.

---

## 🚀 Početak

Prati ove korake za postavljanje okruženja, treniranje modela ili generisanje muzike koristeći već istrenirani *checkpoint*.

### 1. Postavljanje okruženja

Kako bi osigurao da paketi ne dođu u konflikt sa tvojim sistemom, toplo se preporučuje korišćenje virtuelnog okruženja.

**Kreiranje virtuelnog okruženja:**

```bash
python -m venv .venv

```

**Aktivacija virtuelnog okruženja:**

* **Windows:**

```bash
.venv\Scripts\activate

```

* **macOS / Linux:**

```bash
source .venv/bin/activate

```

**Instalacija potrebnih paketa:**

```bash
pip install -r requirements.txt

```

*(Napomena: FluidSynth mora biti instaliran na nivou operativnog sistema kako bi direktno sintetizovao WAV fajlove).*

---

## 🧠 Korišćenje i pokretanje

Ceo proces (*pipeline*) je centralizovan u jednoj Jupyter svesci.

### 2. Otvaranje kontrolnog panela

Otvori **`ssm_main.ipynb`** u svom omiljenom editoru (npr. VSCode ili Jupyter web interfejs).

**Važno:** Obavezno izaberi tačan Python *kernel* koji je povezan sa `.venv` okruženjem kreiranim u prvom koraku.

### 3. Treniranje modela (Opciono)

Ako želiš da treniraš mrežu od nule:

1. Pokreći ćelije u svesci redom.
2. Skripta će izvući podatke iz `music21` korpusa, obraditi ih u kvantizovane nizove vremena/visine tona i započeti PyTorch ciklus treniranja.
3. *Checkpoint*-ovi po epohama će se automatski čuvati u tvom folderu predviđenom za modele.

---

## 🎼 Generisanje muzike

Generisana pesma se nalazi u *static* folderu, pazi da folder ne postane pretrpan.

Ako želiš da preskočiš treniranje i koristiš već istrenirani model, idi na poslednju ćeliju u `ssm_main.ipynb`. Tamo bi trebalo da postoji zakomentarisana linija; otkomentariši liniju sa `best_model.pth` i zakomentariši liniju iznad nje.

Pobrini se da promenljiva `checkpoint_path` pokazuje na tačan direktorijum koji sadrži tvoje `.pth` fajlove.

### Parametri za podešavanje:

Možeš da modifikuješ sledeće promenljive u ćeliji za generisanje kako bi oblikovao izlaz:

* **`checkpoint_epoch`**: Bira stanje modela (npr. `10` često pruža dobar balans između kreativnosti i strukture).
* **`temperature`**: Kontroliše entropiju (nasumičnost) generisanja.
* `0.5` = Konzervativno, strogo pridržavanje naučenih pravila.
* `0.8` = Balansirano i kreativno (Preporučeno).
* `1.0+` = Veoma haotično i nepredvidivo.


* **`target_seconds`**: Željena dužina generisanog audio fajla (npr. `60`).

Pokreni ćeliju da bi autoregresivno komponovao traku, orkestrirao izlaz i renderovao konačni WAV fajl.
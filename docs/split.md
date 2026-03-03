🎵 Noteheads (The "White" vs. "Black")
In standard music notation, "white" (open) and "black" (filled) refer to the rhythmic duration of the noteheads, rather than the accidentals.

E0A2: noteheadWhole (White/Open, whole note)

E0A3: noteheadHalf (White/Open, half note)

E0A4: noteheadBlack (Black/Filled, used for quarter notes, 8th notes, and shorter)

♯♭ Accidentals ("Accids")
Standard accidentals don't typically have white/black variants, but here are the core codes you need to map them:

E260: accidentalFlat (♭)

E261: accidentalNatural (♮)

E262: accidentalSharp (♯)

E263: accidentalDoubleSharp (𝄪)

E264: accidentalDoubleFlat (𝄫)

🏴 Flags (Stem tails)
You mentioned E240—this falls perfectly into the flag category! These attach to the stems of your black noteheads.

E240: flag8thUp (8th note flag, stem pointing up)

E241: flag8thDown (8th note flag, stem pointing down)

E242: flag16thUp (16th note flag, stem up)

E243: flag16thDown (16th note flag, stem down)

🎼 Clefs & Rests
E050: clefG (Treble Clef)

E062: clefF (Bass Clef)

E05C: clefC (Alto/Tenor Clef)

E07A: gClefChange (G clef change)

E07C: fClefChange (F clef change)

E07B: cClefChange (C clef change)

rest32nd: E4E8 "32nd (demisemiquaver) rest"
rest16th: E4E7 "16th (semiquaver) rest"
rest8th: E4E6 "Eighth (quaver) rest"
restQuarter: E4E5 "Quarter (crotchet) rest"
restHalf: E4E4 "Half (minim) rest"
restWhole: E4E3 "Whole (semibreve) rest"
restDoubleWhole: E4E2 "Double whole (breve) rest"
restLonga: E4E1 "Longa rest"
restMaxima: E4E0 "Maxima rest"

⏱️ Time Signatures (timeSig)
In SMuFL, time signatures are generally constructed using individual digits, plus a few special standalone symbols for common meters.

E080: timeSig0 (Digit 0)

E081: timeSig1 (Digit 1)

E082: timeSig2 (Digit 2)

...this continues sequentially up to...

E089: timeSig9 (Digit 9)

E08A: timeSigCommon (Common time, the "C" symbol)

E08B: timeSigCutCommon (Cut time / Alla breve, the "C" with a vertical slash)

(Note: To create a standard 4/4 or 3/4 time signature, notation software generally stacks the respective digit glyphs on top of each other vertically!)

📏 Barlines (barline)
You'll definitely need these to organize those measures.

E030: barlineSingle (Standard single barline)

E031: barlineDouble (Double barline, often used at section changes)

E032: barlineFinal (Final barline, a thin line followed by a thick line at the end of a piece)

E038: barlineDashed (A dashed barline for complex meters or editorial marks)

🔊 Dynamics (dynamic)
These are the standard stylistic letters used to indicate volume. You combine them to make things like mp or ff.

E520: dynamicPiano (p)

E521: dynamicMezzo (m)

E522: dynamicForte (f)

E524: dynamicSforzando (s)

E525: dynamicZ (z — used in sfz)

🎻 Common Articulations (artic)
These are usually placed directly above or below the noteheads. SMuFL often has specific codes depending on whether the mark goes above or below the staff, but here are the standard "above" defaults:

E4A0: articStaccatoAbove (The staccato dot)

E4A4: articTenutoAbove (The tenuto line)

E4A8: articAccentAbove (The standard accent wedge: >)

E4AC: articStaccatissimoAbove (The wedge/teardrop staccato)

E4B0: articMarcatoAbove (The vertical marcato "tent": ^)
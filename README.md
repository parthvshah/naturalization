# Speech Naturalization

An NLP technique to generate natural speech with "uh", "um" and pauses. 

Authors: Parth Shah (parthvipulshah@pesu.pes.edu) and Richa Sharma (richa13sharmaa@gmail.com)

## Installation

Run:

```pip3 install -r requirements.txt```

## Structure

The following files can be run individually with python3. Each program prompts for an input sentence that will be naturalized and outputed to STDOUT.

- bigram.py
- pos_bigram.py
- flow.py
- hybrid.py

## Testing

To run on a corpus of sentences, run ```python3 driver.py``` after changing the input file and method. The method functions are tagged with 'driver' in their respective files. Outputed to STDOUT.

## Resources

Details on the implementation as part of a draft paper and a high level overview as part of a presentation can be found in the 'res' folder.
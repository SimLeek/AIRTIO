#AIRTIO
AI Real Time IO

Methods for converting inputs, such as cameras or ffts, into formats AIs can use

Currently has SPEC, which runs 2D inputs with any number of channels through a laplacian pyramid and outputs tokens once a 'pixel' gains enough energy.
These can be fed into transformers, and should work better for video, theoretically, but this needs to be tested against other formats.

# todo:
* Optimize SPEC to move all pyramid frames into a single rectangle with rectpack and run all conv/mask operations at once
* convert rectpack optimized SPEC to a GPU kernel
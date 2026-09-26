#How to run

Project folder er root e giye ei command dilei hobe :
streamlit run .\app\streamlit_app.py

git bash : streamlit run app/streamlit_app.py

kspace store ta hocche kaggle dataset theke convert kore k-space banano hoyeche

## Compressed sensing lab (extra page)

Same command as above. Streamlit picks up `app/pages/` automatically, so a
second entry "Compressed sensing lab" shows up in the sidebar navigation.

It is the compressed-sensing reconstruction rendered step by step:

1. **The premise** - throw away 95% of the wavelet coefficients of the *fully
   known* image and it still looks the same. This is why the method can work.
2. **One iteration, step by step** - a single FISTA iteration opened up, all
   seven sub-steps with every intermediate array. The "Reveal step by step"
   button animates them in order.
3. **Watch it converge** - the loop running live, frame by frame, with a
   timeline to scrub back through afterwards.
4. **Why random sampling** - the same algorithm on Cartesian / radial /
   variable-density, showing it only works on incoherent artifacts.
5. **The lambda dial** - swept from under- to over-regularised.

Sections 4 and 5 sit behind a button because each runs several full
reconstructions.

### If it breaks, delete it

The lab is three files and nothing else imports them:

    app/pages/1_Compressed_sensing_lab.py    the page
    app/cs_lab_ui.py                         its visualisations
    mri_sim/cs_trace.py                      the instrumented algorithm

Delete those three (and the now-empty `app/pages/`) and the original app is
back exactly as it was - no other file was modified to add this.

`mri_sim/cs_trace.py` runs the same arithmetic as `mri_sim/cs.py`, importing
its helpers rather than copying them. `cs_trace.verify_matches_reference()`
checks the two produce bit-identical images.

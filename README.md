# Info

on this page...
- [Purpose](#purpose)
- [Preliminary results](#preliminary-results)
- [Problem being addressed](#problem-being-addressed)
- [Embeddings -- an overview](#embeddings----an-overview)
- [Future work](#future-work)
- [Other](#other)

---


## Purpose

This is experimental python machine-learning code, to explore using a vision-model to create embeddings to cluster related [Hall-Hoag images][HH] images in the [Brown Digital Repository][BDR].

[HH]: <https://repository.library.brown.edu/studio/item/bdr:9r3a8c4a/>
[BDR]: <https://repository.library.brown.edu/studio/>


## Preliminary results

These are very prelimnary results; this is an outside-of-work side-project.

Nevertheless, the results indicate that this embedding approach could be a very useful tool to group related images.

This [google-spreadsheet][ggl] show clustering results for 50 of the 587 images in the [AFL-CIO][AFL] Hall-Hoag organization. It's useful to skip back and forth from the spreadsheet to the AFL-CIO organization to get a sense of the grouping. The important parts are the last three columns. The technique I used (which I describe below) offers the ability to adjust _lots_ of settings. 

The columns "01-groups" and "02-groups" show the grouping output from my code by making a single change to one of numerous settings. The column "real-groups" shows the actual groups from a manual inspection of the images.

The alternating light-yellow and light-red cells highlight groupings of the "real" images for comparison with the 2 prediction columns. (A clarification note: Look at the last grouping... Note that it doesn't matter that the "real" column has the number "10" for all the related images, but the other two prediction-columns have the numbers "15" and "12". The important thing is the grouping, not the actual numbers.)

[ggl]: <https://docs.google.com/spreadsheets/d/10_lqr7n4qQ2e0zgZxXNt4rKIkLiTMlLg7UoLxF2G7Wc/>
[AFL]: <https://repository.library.brown.edu/studio/item/bdr:9r3a8c4a/>


## Problem being addressed

Context: a Hall-Hoag organization, like this [AFL-CIO organization][AFL], can have hundreds or thousands of scanned-images. There might be two-pages for a flyer (front and back), followed by four-pages for a letter, followed by 15-pages for a report. But all of those pages are just single-page images with zero item-level-metadata (not even a title). They're not "grouped" as a flyer, a letter, a report -- in any way.

For very good reasons we're ingesting these this way -- but obviously we'd eventually like to be able to make improvements. Because we'll end up having some 900,000 scanned images, we'll be looking at programmatic ways to create metadata and group related images. Thus this preliminary experiment,  to get a feel for how "embeddings" can be used to cluster related images.


## Embeddings -- an overview

I've mentioned that this clustering technique uses a vision-model to create "embeddings". An explanation...

Imagine you're filling out a detailed checklist for a series of images of dogs. Imagine the checklist includes things like:

- Height (short, medium, tall)
- Fur length (short, medium, long)
- Ear shape (pointy, floppy, round)
- Color (black, brown, white, spotted, etc.)
- Snout length (short, medium, long)

Each dog-picture gets a unique set of answers, and you can compare dogs by how many checklist items match.

You could say things like "Height" and "Fur" are "categories" or "features", and a "value" is entered for each feature.

It makes sense that the more features you track, the more detailed your understanding of the dog image.

Now, instead of entering words for the checklist-values, imagine that each of these features gets a number instead. 

This is similar to what a vision-model does:

- Instead of a few features like color and ear-shape, a vision-model can track hundreds or even thousands of subtle details. (Those "features" are called "dimensions" in machine-learning jargon.)

- Instead of writing them on paper, a vision-model stores its evaluation as a long list of numbers. (Called an "embedding vector" in machine-learning jargon.)

- Instead of checking which words match, we can examine those numbers. By comparing them mathematically, we can determine which images are most similar — closer numbers in the same dimensions mean more similarity.


## Future work

Regarding this embedding technique, there's lots more experimenting that could be done:

- I could try different vision-models.
- I could try different versions of a given vision model (I chose a somewhat smaller one favoring speed over accuracy).
- There are different settings I could change in the process of creating the numeric-representations of each image.
- There are different settings I could change in the process of evaluating similarity.
- For speed of processing, I'm using images that are only one-fourth the full-size images. Perhaps the higher-resolution images would yield better results.
- I'm only passing images to the vision-model. I could integrate a parallel process of passing OCRed text to a model (or instructing a multi-modal model to read the text) -- and create embeddings from that, as well.

Likely the best clustering results will involve using a variety of techniques, not only embeddings. There will be many situations where programmtic grouping is likely to prove challenging. A typical example would be where the colors and design of a "cover-page" will be very different from more predictable "article-pages" that follow. Still, this could be a very valuable "initial-pass" that by itself, and certainly with human followup, could facilitate useful grouping.


## Usage

Assuming the wonderful python package `uv` is installed ([info link][uv]):

```
$ uv run ./cc__hh_ml_code.py
```

[uv]: <https://docs.astral.sh/uv/>


## Other

- One oddity of the code is that it stores all the vector-arrays (lists of numbers) to an sqlite database -- but then reads all that data into memory at the "determine-similarity" step. That's because sqlite doesn't natively support vector-searching. For the number of images I was dealing with in this experiment, that's not a problem -- but it'd be worth  investigating sqlite extensions that would allow direct db-vector-searches, or using a db that supports them.

- A sincere thanks to Brendan Quinn and Michael B. Klein of the Northwestern University Library for their terrific full-day [code4lib-con-2024 full-day LLM-workshop][c4l-llm] that really solidified my sense of the concept of embeddings, as well as how to work with them programmatically. 

[c4l-llm]: <https://2024.code4lib.org/workshop/GenAI4Lib-A-Practical-Introduction-to-Large-Language-Models>

---

(end)
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/charliemalach/BIOCHEM-Data-Mining">
    <img src="./image/logo.png" alt="Logo" width="100" height="125">
  </a>

  <h3 align="center">Bio Chem Data Mining</h3>

  <p align="center">
    An exhaustive overview of the Bio Chem Data Mining Application
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

![Map of Data][map]

Our project’s main focus was looking into the differences between Valparaiso’s
Biochemistry program and comparisons to the biochemistry programs at other colleges and
universities. Most importantly being schools with accredited biochemistry programs. To check
whether a school has an accredited biochemistry program, we refer to the American Society for
Biochemistry and Molecular Biology. For our purposes, we focused mainly on what differences
Valparaiso could include in its biochemistry program. This same type of analysis could be run on different schools, and also on different programs other than biochemistry. Another key thing we took into consideration is that Valparaiso’s biochemistry program could, in addition to being different from accredited biochemistry programs, differ from general non-accredited biochemistry programs. If further expanded upon, this work could be used to improve academic programs. Theoretically, the use of this program could also help students accurately pick which school they want to attend based on the content of offered courses. Our learning objectives for this project prompted us to use Topic Modeling to extract important topics from sentences, and then be able to operationally define similarities and differences when it comes to comparing lists of topics. Using these definitions, we aimed to perform a statistical analysis between Valparaiso and accredited and non-accredited schools. We hypothesized that Valparaiso’s biochemistry
was different from most accredited universities, and more akin to non-accredited universities

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

These are the tools we utilized throughout the implementation of our application:

- [![python][python-img]][python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Usage

We ended up making a program that can create an LDA model that outputs a list of
topics, as well as the TFIDF of course descriptions so that we could see how. We considered
the terms used in Valparaiso University courses. Then, we looked at the terms that appeared
more than once and used that to compare with the lists of important terms from the topics we
generated from accredited and non-accredited schools. Because there were more of Valpo’s
terms found in the list for non-accredited schools, we found that Valpo’s biochemistry program
had more in common with the biochemistry programs of non accredited schools. This supports
our original hypothesis that although Valpo’s biochemistry was different from accredited and
non-accredited schools’ programs, it had more in common with non-accredited schools.

![Main Application][main-screenshot]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [x] We want to use topic modeling extract important topics from sentences
- [x] We want to able to operationally define similar and different when it comes to comparing lists of topics
- [x] Using these definitions, we aim to perform statistical analysis between Valparaiso, and accredited and non-accredited schools

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Charlie Malachinski (Software Developer) - charlie.malachinski@valpo.edu

Project Link: [https://github.com/charliemalach/BIOCHEM-Data-Mining](https://github.com/charliemalach/BIOCHEM-Data-Mining)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Nltk](https://www.nltk.org/)
- [Regular Expression (re)](https://docs.python.org/3/library/re.html)
- [Numpy](https://numpy.org/)
- [Matplotlib.pylplot](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [LDA (LatentDirichletAllocation)](http://scikit-learn.sourceforge.net/stable/modules/generated/sklearn.lda.LDA.html)
- [pyLDAvis](https://github.com/bmabey/pyLDAvis)
- [Warnings](https://docs.python.org/3/library/warnings.html)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[map]: ./image/map.jpg
[main-screenshot]: ./image/main.jpg
[python-img]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/

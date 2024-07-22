# LaTeX thesis git

## Basics

Before downloading / cloning this git or even worse - directly starting with your TeX-_code_,
please take a step back and have a look at our guidelines, tutorials and best-practise examples:

 - [wiki - latex hints](https://wiki.lsr.ei.tum.de/info/latex_hints)
 - [Best practice for thesis writing - staff version](https://wiki.lsr.ei.tum.de/groups/murola/thesiswriting)
 - [Best practice for thesis writing - student version](https://wiki.lsr.ei.tum.de/thesiswriting_students)
 - [internal repo wiki](https://git.lsr.ei.tum.de/general/latex-student-thesis-template/wikis/home) __not up to date__

If you do not have LaTeX installed yet, please check either the [link above at our wiki](https://wiki.lsr.ei.tum.de/info/latex_hints) or the
[installation-page](https://git.lsr.ei.tum.de/general/latex-student-thesis-template/wikis/installation).
In case you do not/ cannot install LaTex, [TUM's version of Sharelatex](https://sharelatex.tum.de) is an alternative for you, but __ASK YOUR SUPERVISOR FIRST, IF HE IS FINE WITH THAT!!__

After Downloading / cloning this repo please be aware of __reading through the included tutorials carefully__ as it
directly depicts a majority of the most common problems but also errors when struggling with LaTeX.

## Hints for intermediate to advanced LaTeX-users

These pages might be of a help for you
- [Use of English Titles](http://www.tum.de/fileadmin/w00bfo/www/Studium/Dokumente/Pruefungsangelegenheiten/Verwendung_des_Englischen_in_Thesistiteln.pdf)
- [Latex Symbol Finder](http://detexify.kirelabs.org/classify.html)
- [Bibliography-style-comparison](https://de.sharelatex.com/learn/Bibtex_bibliography_styles)
- [Online equation editor](https://www.codecogs.com/latex/eqneditor.php) for fast eqation checks / compilation

For 16:9 ratio simply use `\documentclass[aspectratio=169]{beamer}`. This works fine with the template

### Use presentation mode for LaTeX presentation

You may use [pympress](https://pympress.readthedocs.io/en/latest/README.html), which provides a
presenter mode similar to powerpoint

![pympress_example](https://cimbali.github.io/pympress/resources/pympress-screenshot.png)

please follow the installation instructions via the link above, as noting them down here will just lead
to backward incompatibility.

In order to use pympress within this template, simply add the option
```presentermode``` to the flags on the top-level of your presentation.

Now, you can either compile the PDF to your liking (TexStudio, ```make```, etc.) and then open the PDF via 

```bash
python3 -m pympress main.pdf
```

or simply run 

```bash
latexmk main
```

from a terminal, as this will launch pympress after build, and update the files once you initiate a file change.

## Last but not least

 1. __Do spellcheck__
 2. Do additional checks by having a look on some guidelines, e.g. in our [wiki](https://wiki.lsr.ei.tum.de/thesiswriting_students)
 3. __If you did not do 1., you will not get any feedback from your supervisor!__

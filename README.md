# Foreground detection

## Demo:

* Run file main.py to demo or evaluate

```
    python main.py
```

## Dataset

* [CDnet2012](https://drive.google.com/file/d/1_wl5nv9zGHDut2sVQE6gqCT5ThkR4pYB/view?usp=sharing)

* [CDnet2014](https://drive.google.com/file/d/1-X5USDsGS0NQhgCPz9Q0dRuX8q_uSNCo/view?usp=sharing)

## Docs:

* [Silde](https://www.canva.com/design/DAFEQ2n5LT4/e-EP0Hao-jAr4CLDkSztrw/edit?utm_content=DAFEQ2n5LT4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton&fbclid=IwAR3KUsOTEQ_JPLL4jLOLiTuIzX1V96MfSLxakCKjPxdL1qoLFHtheMkUc58)

## Evaluation 

* We have benchmark the 5 Algorithm which has already implemented in PYBGS lib(C++ code), Vibe_python(Re-implement base on psudo code), Vibe_fast_python(Base on C++ source):
    * Independent Multimodal
    * PixelBasedAdaptiveSegmenter(PBAS)
    * SuBSENSE
    * PAWCS 
    * ViBe(C++ pybgs) (22,70 fps)
    * ViBe(Python base on psudo code) ( 0.26 fps)
    * ViBe(Python optimize base on C++ source) (10.21 fps)
* Note: FPS is benchmark base on Tramstop video(intermittent Object Motion challenge in CDnet2012 dataset). 
* Evaluate detail(Performance and FPS) : [Sheet](https://docs.google.com/spreadsheets/d/1EirdhREqFQWfEWerQnEbBNRvsPu_Ab1o/edit?usp=sharing&ouid=100934651474655278247&rtpof=true&sd=true)

## References

* [ViBE Algorithm](http://www.telecom.ulg.ac.be/publi/publications/barnich/Barnich2011ViBe/index.html)
* [Awesome background subtraction](https://github.com/murari023/awesome-background-subtraction)
* [BGS Lib](https://github.com/andrewssobral/bgslibrary)
* [Background subtraction Website by Thierry BOUWMANS](https://sites.google.com/site/backgroundsubtraction/overview)
* [Change detection challenges(CDNET)](https://web.archive.org/web/20220524170604/http://www.changedetection.net/)
#' Knit .Rmd file into a jekyll blog post 
#'  
#' Code modified from https://github.com/jfisher-usgs/jfisher-usgs.github.com/blob/master/R/KnitPost.R
#' Also includes ideas from https://github.com/supstat/vistat/blob/gh-pages/_bin/knit
#' @export
#' @examples
#' \dontrun{library(devtools); install_github('knitPost', 'cpsievert')}
#' library(knitPost)
#' setwd("~/Desktop/github/local/cpsievert.github.com/")
#' knitPost("2013-05-15-hello-jekyll.Rmd")
input <- commandArgs(trailingOnly = TRUE)
knitPost <- function(post="1-example.Rmd", baseUrl="{{ site.baseurl }}/") {
  
  require(knitr)
  
  sourcePath <- file.path(getwd(), post)
  opts_knit$set(base.url=baseUrl)
  base <- sub("\\.[Rr]md$", "", basename(sourcePath))
  fig.path <- paste0(baseUrl,"images/")
  opts_chunk$set(fig.path=fig.path)
  opts_chunk$set(fig.cap="center")
  render_jekyll()
  knit(sourcePath, paste('../_posts/', base, '.md', sep = ''), envir=parent.frame())
}

knitPost(input)

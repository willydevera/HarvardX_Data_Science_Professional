A new repository to add the required PH125_9x Capstone Movielens Project reports in RMD and PDF formats, as well as the main script/code in R format that generates the predicted movie ratings, RMSE scores, and corresponding grades from the 18 models that were built.

Heads-up:

It may take forever if you would run the varimp_earth code chunk in Section 2.2.5 - Variable importance from the RMD file. Or you may run into memory allocation error messages or so.  That was the reason this code chunk was set to eval=FALSE.  To be able to generate variable importance from the RMD file, I suggest that you first generate the earth_edx file by running Section 3.5 - Check variable importance - from Main R code, then save it as earth_edx.RDA using the command:
save(earth_edx, file="{your_folder_path}\\earth_edx.rda")

And then run the alternative_load_earth_edx code chunk in Section 2.2.5 - Variable importance from the RMD file. Note the path where you saved it. The earth_edx file is about 205 MB and this cannot be uploaded to github.

As well, running the stats linear model and h2o random forests, generalized linear model (glm), deep neural network, and gradient boosting machine (gbm) models may may take a longer while and require larger memory.  In fact, the time to knit this rmd file would require about 7-8 hours on R 3.5.x Windows 10, 64-bit running on Intel Core i7-7700 with 32 GB RAM.

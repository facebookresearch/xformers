set(DIR_SRCS)

##
### Find all sources in this directory.
##

file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

# Add the sources to the build.
set(SPUTNIK_SRCS ${SPUTNIK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

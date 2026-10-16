#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include "moonbit.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef char llmcalc_requires_64_bit_off_t[sizeof(off_t) >= 8 ? 1 : -1];

/* Only OS I/O lives in C; parsing, limits, and arithmetic live in MoonBit. */
int32_t llmcalc_open_readonly(moonbit_bytes_t path) {
  int fd;
  do { fd = open((const char *)path, O_RDONLY | O_CLOEXEC | O_NONBLOCK); }
  while (fd < 0 && errno == EINTR);
  return fd < 0 ? -errno : fd;
}

int64_t llmcalc_file_size(int32_t fd) {
  struct stat st;
  if (fstat(fd, &st) != 0) return -errno;
  if (!S_ISREG(st.st_mode)) return -EINVAL;
  if (st.st_size < 0) return -EOVERFLOW;
  return (int64_t)st.st_size;
}

int32_t llmcalc_pread_exact(int32_t fd, int64_t position,
                          moonbit_bytes_t buffer, int32_t length) {
  if (length < 0 || length > Moonbit_array_length(buffer) || position < 0 ||
      position > INT64_MAX - length) return -EINVAL;
  int32_t done = 0;
  while (done < length) {
    ssize_t count = pread(fd, buffer + done, (size_t)(length - done),
                          (off_t)(position + done));
    if (count < 0) {
      if (errno == EINTR) continue;
      return -errno;
    }
    if (count == 0) break;
    done += (int32_t)count;
  }
  return done;
}

int32_t llmcalc_close_file(int32_t fd) {
  /* Do not retry close after EINTR: a reused descriptor must not be closed. */
  return close(fd) == 0 ? 0 : -errno;
}

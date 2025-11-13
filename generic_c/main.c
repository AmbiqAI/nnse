#include "AudioPipe_wrapper.h"
#include <stdint.h>
#include <stdio.h>
#include "kiss_fftr.h"

int16_t pcm_output[512];

#define BUFFER_SIZE 172480
extern int16_t input_data[BUFFER_SIZE];  // 10ms @ 16kHz

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Use #pragma pack(push, 1) or similar if your compiler
// aligns structure members on different boundaries by default
typedef struct {
    char chunkId[4];
    uint32_t chunkSize;
    char format[4];
    char subchunk1Id[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2Id[4];
    uint32_t subchunk2Size;
} WavHeader;
#define SAMPLES_PER_READ 160
int16_t buffer[SAMPLES_PER_READ];
size_t elements_read;
int main()
{
    FILE *infile = fopen("wavs/test.wav", "rb");
    if (infile == NULL) {
        // Handle error
        return 1;
    }

    WavHeader header;
    if (fread(&header, sizeof(WavHeader), 1, infile) != 1) {
        // Handle error
        fclose(infile);
        return 1;
    }

    // Validation checks (e.g., check 'RIFF', 'WAVE', 'fmt ', 'data' IDs, format values)
    if (header.numChannels != 1 || header.bitsPerSample != 16 || header.audioFormat != 1) {
        printf("Unsupported WAV format: requires mono 16-bit PCM.\n");
        fclose(infile);
        return 1;
    }
    // ... also check for "data" subchunk, as extra chunks might exist before it
    // For simplicity here, we assume the data follows the standard header immediately.
    

    // Loop through the data section, reading 160 samples at a time
    
    AudioPipe_wrapper_init();
    AudioPipe_wrapper_reset();
    FILE *fout = fopen("wavs/output.pcm", "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output.pcm for writing\n");
        return 1;
    }

    int i = 0;
    while ((elements_read = fread(buffer, sizeof(int16_t), SAMPLES_PER_READ, infile)) > 0) {
        // Process the 'elements_read' number of samples in the buffer
        printf("Processed 160 samples. Frame number: %d\n", i);
        i++;
        if (elements_read < SAMPLES_PER_READ) {
            // Reached the end of the file/data chunk (last chunk might be smaller)
            break;
        } else {
            AudioPipe_wrapper_frameProc(buffer, pcm_output);
            fwrite(pcm_output, sizeof(int16_t), 160, fout);
        }
    }
    AudioPipe_wrapper_free();
    // Close the file
    fclose(infile);
    fclose(fout);

    return 0;
}

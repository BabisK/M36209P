from nltk import downloader

def main():
    downloader.download('europarl_raw')
    from nltk.corpus import europarl_raw

if __name__ == "__main__":
    main()
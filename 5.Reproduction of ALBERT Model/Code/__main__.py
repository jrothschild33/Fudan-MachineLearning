# coding: utf8

def main():
    import sys
    if sys.argv[1] == "convert":
        from transformers.commands import convert
        convert(sys.argv)
    elif sys.argv[1] == "train":
        from transformers.commands import train
        train(sys.argv)
    elif sys.argv[1] == "serve":
        pass

if __name__ == '__main__':
    main()

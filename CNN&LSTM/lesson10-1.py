# coding: utf-8
# captcha

from captcha.image import ImageCaptcha
import random
import sys

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    # captcha = image.generate(captcha_text)
    image.write(captcha_text, 'captcha/images/' + captcha_text + '.jpg')


if __name__ == '__main__':
    num = 10000
    # 实际上没有10000个，因为存在重复的
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write("\r>> Creating image %d/%d" % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print('生成完毕')

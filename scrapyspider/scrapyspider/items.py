# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


# class ScrapyspiderItem(scrapy.Item):
#     # define the fields for your item here like:
#     job_name = scrapy.Field()
#     job_company = scrapy.Field()
#     job_address = scrapy.Field()
#     job_salary = scrapy.Field()
#     job_date = scrapy.Field()
#     job_url = scrapy.Field()
#     job_info = scrapy.Field()
#     job_type = scrapy.Field()

class douban_item(scrapy.Item):
    name = scrapy.Field()
    director = scrapy.Field()
    rate = scrapy.Field()
    genres = scrapy.Field()
    page_url = scrapy.Field()
    pic_url = scrapy.Field()
    region = scrapy.Field()
    imdb = scrapy.Field()
    brief = scrapy.Field()
    targets = scrapy.Field()



# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from .ohmy_DB.douban_db import douban_db


class ScrapyspiderPipeline:

    def open_spider(self, spider):
        self.db = douban_db(config='./database.json', logger=spider.logger)
        self.logger = spider.logger
        self.number = 0
        self.storage = []

    def process_item(self, item, spider):
        self.number += 1
        print(f'got {self.number}', item)
        if item['targets'] != 'NULL':
            self.storage.append(item)
        else:
            self.insert_to_db(item)

        return item

    def insert_to_db(self, item):
        params = [
            item['name'], item['director'], item['rate'],
            item['genres'], item['page_url'], item['pic_url'], item['region'],
            item['imdb'], item['brief'], item['targets']
        ]
        self.db.insert_item(params)

    def close_spider(self, spider):
        for item in self.storage:
            if item['targets'].isspace():
                item['targets'] = 'NULL'
            self.insert_to_db(item)
        self.db.close()

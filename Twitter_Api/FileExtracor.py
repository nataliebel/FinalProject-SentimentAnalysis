
class FileExtracor():

    def extract_file(self, file):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = [x.strip() for x in f]
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return content

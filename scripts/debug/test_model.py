from src.pipelines.factory import ModelFactory
res = ModelFactory().translate("Gommapiuma coibentata", "ita_Latn", "eng_Latn")
print(res)
query for last 15 day record
-----------------------------
db.datecol.find(
{
    "date": 
    {
        $gte: new Date((new Date().getTime() - (15 * 24 * 60 * 60 * 1000)))
    }
}
).sort({ "date": -1 })


query for fetch record between two dates:
-----------------------------------------
const startDate = new Date(2021 - 06 - 06);
const endDate = Date.now(2021 - 06 - 08);

if (startDate === '' || endDate === '') {
  return res.status(400).json({
    msg: 'Please ensure you pick two dates',
  });
}

const allRecord = await Record.find({
  user: req.user.id,
  date: {
    $gte: new Date(new Date(startDate).setHours(0, 0, 0)),
    $lt: new Date(new Date(endDate).setHours(23, 59, 59)),
  },
});